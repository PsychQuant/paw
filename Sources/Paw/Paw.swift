import ArgumentParser

@main
struct Paw: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "paw",
        abstract: "Swift-native LLM inference server for Apple Silicon.",
        version: "0.1.0",
        subcommands: [
            ServeCommand.self,
            PullCommand.self,
            ModelsCommand.self,
            RemoveCommand.self,
        ],
        defaultSubcommand: ServeCommand.self
    )
}
