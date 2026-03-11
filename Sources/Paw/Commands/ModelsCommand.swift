import ArgumentParser
import Foundation

struct ModelsCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "models",
        abstract: "List all downloaded MLX models."
    )

    func run() async throws {
        let models = ModelScanner.scanAvailableModels()

        if models.isEmpty {
            print("No downloaded models found.")
            print("Download one with: paw pull <model-id>")
            return
        }

        // Header
        let idWidth = max(50, models.map(\.id.count).max()! + 2)
        let header = "MODEL".padding(toLength: idWidth, withPad: " ", startingAt: 0)
            + "SIZE"
        let separator = String(repeating: "─", count: idWidth + 12)

        print(header)
        print(separator)

        for model in models {
            let sizeStr = formatBytes(model.sizeBytes)
            let line = model.id.padding(toLength: idWidth, withPad: " ", startingAt: 0)
                + sizeStr
            print(line)
        }

        print()
        print("\(models.count) model(s) found.")
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
