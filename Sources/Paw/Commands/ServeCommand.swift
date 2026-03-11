import ArgumentParser
import Foundation
import Logging
import Vapor

struct ServeCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "serve",
        abstract: "Start the OpenAI-compatible inference server."
    )

    @ArgumentParser.Option(name: .long, help: "Model ID to preload (e.g. mlx-community/Qwen3-4B-4bit).")
    var model: String?

    @ArgumentParser.Option(name: .long, help: "Host address to bind.")
    var host: String = "127.0.0.1"

    @ArgumentParser.Option(name: .long, help: "Port to listen on.")
    var port: Int = 8080

    @ArgumentParser.Flag(name: .long, help: "Enable Vision Language Model mode.")
    var vlm: Bool = false

    @ArgumentParser.Option(name: .long, help: "Embedding model path or HuggingFace ID.")
    var embeddingModel: String?

    @ArgumentParser.Flag(name: .long, help: "Enable prompt caching.")
    var enablePromptCache: Bool = false

    @ArgumentParser.Option(name: .long, help: "Prompt cache size in MB.")
    var promptCacheSizeMb: Int = 1024

    @ArgumentParser.Option(name: .long, help: "Prompt cache TTL in minutes.")
    var promptCacheTtlMinutes: Int = 30

    func run() async throws {
        print("🐾 Paw — MLX Inference Server")
        print("   Host: \(host)")
        print("   Port: \(port)")
        print("   VLM:  \(vlm)")

        // Configure Vapor
        var env = try Environment.detect()
        try LoggingSystem.bootstrap(from: &env)
        let logger = Logger(label: "paw.server")

        let app = try await Application.make(env)
        defer { Task { try? await app.asyncShutdown() } }

        app.http.server.configuration.hostname = host
        app.http.server.configuration.port = port

        // CORS
        let corsConfig = CORSMiddleware.Configuration(
            allowedOrigin: .all,
            allowedMethods: [.GET, .POST, .OPTIONS, .DELETE],
            allowedHeaders: [
                .accept, .authorization, .contentType, .origin,
                .xRequestedWith,
            ]
        )
        app.middleware.use(CORSMiddleware(configuration: corsConfig))

        // Create model manager
        let modelManager = ModelManager(
            defaultModelPath: model,
            isVLM: vlm,
            logger: logger
        )

        // Create embedding model manager
        let embeddingModelManager = EmbeddingModelManager(
            defaultModelId: embeddingModel,
            logger: logger
        )

        // Prompt cache (optional)
        let promptCacheManager: PromptCacheManager?
        if enablePromptCache {
            promptCacheManager = PromptCacheManager(
                maxSizeMB: promptCacheSizeMb,
                ttlMinutes: promptCacheTtlMinutes,
                logger: logger
            )
            print(
                "   Prompt Cache: enabled (\(promptCacheSizeMb)MB, TTL \(promptCacheTtlMinutes)min)"
            )
        } else {
            promptCacheManager = nil
        }

        // Preload model if specified
        if let modelId = model {
            print("   Preloading model: \(modelId)...")
            _ = try await modelManager.getModel(requestedModelId: modelId)
            print("   Model loaded successfully.")
        }

        // Register all routes
        try await configureRoutes(
            app,
            modelManager: modelManager,
            embeddingModelManager: embeddingModelManager,
            isVLM: vlm,
            promptCacheManager: promptCacheManager
        )

        print("   Server starting on http://\(host):\(port)")
        try await app.execute()
    }
}
