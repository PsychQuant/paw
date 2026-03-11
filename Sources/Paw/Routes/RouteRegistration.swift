import Foundation
import Vapor

/// Register all API routes for the Paw server.
func configureRoutes(
    _ app: Application,
    modelManager: ModelManager,
    embeddingModelManager: EmbeddingModelManager,
    isVLM: Bool,
    promptCacheManager: PromptCacheManager?
) async throws {
    try registerCompletionsRoute(
        app, modelManager: modelManager, promptCacheManager: promptCacheManager)
    try registerChatCompletionsRoute(
        app, modelManager: modelManager, isVLM: isVLM,
        promptCacheManager: promptCacheManager)
    registerEmbeddingsRoute(app, embeddingModelManager: embeddingModelManager)
    registerModelsRoute(app, modelManager: modelManager)
    registerCacheManagementRoutes(app, promptCacheManager: promptCacheManager)
}

/// Register cache management endpoints.
func registerCacheManagementRoutes(
    _ app: Application, promptCacheManager: PromptCacheManager?
) {
    app.get("v1", "cache", "status") { req async throws -> CacheStatusResponse in
        req.logger.info("Handling /v1/cache/status request")

        guard let cacheManager = promptCacheManager else {
            return CacheStatusResponse(
                enabled: false,
                entryCount: 0,
                currentSizeMB: 0,
                maxSizeMB: 0,
                ttlMinutes: 0,
                stats: CacheStatsResponse(
                    hits: 0,
                    misses: 0,
                    evictions: 0,
                    hitRate: 0,
                    totalTokensReused: 0,
                    totalTokensProcessed: 0,
                    averageTokensReused: 0
                )
            )
        }

        let stats = await cacheManager.getStats()
        let status = await cacheManager.getCacheStatus()
        let maxSizeMB = await cacheManager.maxCacheSizeMB
        let ttlMinutes = await cacheManager.cacheTTLMinutes

        return CacheStatusResponse(
            enabled: true,
            entryCount: status.entryCount,
            currentSizeMB: status.sizeMB,
            maxSizeMB: maxSizeMB,
            ttlMinutes: ttlMinutes,
            stats: CacheStatsResponse(
                hits: stats.hits,
                misses: stats.misses,
                evictions: stats.evictions,
                hitRate: stats.hitRate,
                totalTokensReused: stats.totalTokensReused,
                totalTokensProcessed: stats.totalTokensProcessed,
                averageTokensReused: stats.averageTokensReused
            )
        )
    }

    app.delete("v1", "cache") { req async throws -> CacheClearResponse in
        req.logger.info("Handling DELETE /v1/cache request")

        guard let cacheManager = promptCacheManager else {
            return CacheClearResponse(
                success: false,
                message: "Prompt cache is not enabled"
            )
        }

        await cacheManager.clearCache()

        return CacheClearResponse(
            success: true,
            message: "Cache cleared successfully"
        )
    }
}
