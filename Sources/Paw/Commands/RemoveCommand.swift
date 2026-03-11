import ArgumentParser
import Foundation

struct RemoveCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "remove",
        abstract: "Remove a downloaded model."
    )

    @ArgumentParser.Argument(help: "Model ID to remove (e.g. mlx-community/Qwen3-4B-4bit).")
    var modelId: String

    @ArgumentParser.Flag(name: .long, help: "Skip confirmation prompt.")
    var force: Bool = false

    func run() async throws {
        let models = ModelScanner.scanAvailableModels()

        guard let model = models.first(where: { $0.id == modelId }) else {
            print("Error: Model '\(modelId)' not found.")
            print("Run `paw models` to see downloaded models.")
            throw ExitCode.failure
        }

        if !force {
            print("Remove model '\(modelId)'?")
            print("  Path: \(model.path)")
            print("  Size: \(formatBytes(model.sizeBytes))")
            print()
            print("Type 'yes' to confirm: ", terminator: "")
            fflush(stdout)

            guard let answer = readLine(), answer.lowercased() == "yes" else {
                print("Cancelled.")
                return
            }
        }

        do {
            try FileManager.default.removeItem(atPath: model.path)
            print("Removed: \(modelId)")
        } catch {
            print("Error removing model: \(error.localizedDescription)")
            throw ExitCode.failure
        }
    }

    private func formatBytes(_ bytes: UInt64) -> String {
        let gb = Double(bytes) / 1_073_741_824
        if gb >= 1.0 {
            return String(format: "%.1f GB", gb)
        }
        let mb = Double(bytes) / 1_048_576
        return String(format: "%.0f MB", mb)
    }
}
