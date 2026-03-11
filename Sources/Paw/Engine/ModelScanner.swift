import Foundation

/// Information about a downloaded model discovered on disk.
struct ScannedModel {
    let id: String
    let path: String
    let sizeBytes: UInt64
}

/// Scans the HuggingFace cache directory for valid MLX model directories.
enum ModelScanner {

    /// Default HuggingFace cache directory on macOS.
    static let defaultCacheDir: String = {
        let home = FileManager.default.homeDirectoryForCurrentUser.path
        return "\(home)/Documents/huggingface/models"
    }()

    /// Scan for all valid MLX models in the cache directory.
    ///
    /// A valid model directory must contain:
    /// - config.json
    /// - A tokenizer file (tokenizer.json or tokenizer_config.json)
    /// - At least one .safetensors weight file
    static func scanAvailableModels(cacheDir: String? = nil) -> [ScannedModel] {
        let baseDir = cacheDir ?? defaultCacheDir
        let fileManager = FileManager.default

        guard fileManager.fileExists(atPath: baseDir) else {
            return []
        }

        var results: [ScannedModel] = []

        // HuggingFace cache structure: models/<org>/<repo>/...
        guard let orgDirs = try? fileManager.contentsOfDirectory(atPath: baseDir) else {
            return []
        }

        for org in orgDirs where !org.hasPrefix(".") {
            let orgPath = "\(baseDir)/\(org)"
            var isDir: ObjCBool = false
            guard fileManager.fileExists(atPath: orgPath, isDirectory: &isDir),
                  isDir.boolValue else { continue }

            guard let repoDirs = try? fileManager.contentsOfDirectory(atPath: orgPath) else {
                continue
            }

            for repo in repoDirs where !repo.hasPrefix(".") {
                let repoPath = "\(orgPath)/\(repo)"
                guard fileManager.fileExists(atPath: repoPath, isDirectory: &isDir),
                      isDir.boolValue else { continue }

                if isValidMLXModel(at: repoPath) {
                    let modelId = "\(org)/\(repo)"
                    let size = directorySize(at: repoPath)
                    results.append(ScannedModel(id: modelId, path: repoPath, sizeBytes: size))
                }
            }
        }

        return results.sorted { $0.id < $1.id }
    }

    /// Check whether a directory contains the required files for an MLX model.
    private static func isValidMLXModel(at path: String) -> Bool {
        let fileManager = FileManager.default

        // Must have config.json
        guard fileManager.fileExists(atPath: "\(path)/config.json") else {
            return false
        }

        // Must have a tokenizer file
        let hasTokenizer = fileManager.fileExists(atPath: "\(path)/tokenizer.json")
            || fileManager.fileExists(atPath: "\(path)/tokenizer_config.json")
        guard hasTokenizer else {
            return false
        }

        // Must have at least one .safetensors file
        guard let contents = try? fileManager.contentsOfDirectory(atPath: path) else {
            return false
        }
        let hasSafetensors = contents.contains { $0.hasSuffix(".safetensors") }

        return hasSafetensors
    }

    /// Calculate the total size of a directory in bytes.
    private static func directorySize(at path: String) -> UInt64 {
        let fileManager = FileManager.default
        guard let enumerator = fileManager.enumerator(atPath: path) else {
            return 0
        }

        var totalSize: UInt64 = 0
        while let file = enumerator.nextObject() as? String {
            let filePath = "\(path)/\(file)"
            if let attrs = try? fileManager.attributesOfItem(atPath: filePath),
               let size = attrs[.size] as? UInt64 {
                totalSize += size
            }
        }
        return totalSize
    }
}
