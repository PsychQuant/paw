import ArgumentParser
import Foundation
import MLXLMCommon
import MLXLLM

struct PullCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "pull",
        abstract: "Download a model from HuggingFace Hub."
    )

    @ArgumentParser.Argument(help: "HuggingFace model ID (e.g. mlx-community/Qwen3-4B-4bit).")
    var modelId: String

    func run() async throws {
        print("Pulling model: \(modelId)")
        print("This may take a while for large models...")

        let modelConfig = ModelConfiguration(id: modelId)

        let container = try await LLMModelFactory.shared.loadContainer(
            configuration: modelConfig
        ) { progress in
            let percent = Int(progress.fractionCompleted * 100)
            print("\r  Downloading... \(percent)%", terminator: "")
            fflush(stdout)
        }

        print()
        print("Model downloaded successfully.")
        print("  ID: \(modelId)")
        print("  Path: \(await container.configuration.name)")
        print()
        print("Start serving with:")
        print("  paw serve --model \(modelId)")
    }
}
