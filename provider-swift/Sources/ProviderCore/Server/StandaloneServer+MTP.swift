import Foundation

extension StandaloneServer {
    func specDecPreparation(
        modelId: String, modelInfo: ModelInfo, modelDirectory: URL? = nil
    ) async -> SpecDecPreparation {
        // The shared funnel validates embedded Qwen heads and the external
        // Gemma assistant. Standalone mode only consults local artifacts;
        // missing assistant bytes preserve target-only serving.
        let inlineDeclaration = modelDirectory.map {
            SpecDecStore.inlineDeclarationProbe(directory: $0)
        } ?? .absent
        return await specDecFunnel.prepare(
            .init(
                modelId: modelId,
                modelType: modelInfo.modelType,
                enabled: config.mtpMode.enablesMTP(
                    forModelType: modelInfo.modelType,
                    embeddedArtifactDeclared: inlineDeclaration.mayDeclareEmbeddedArtifact,
                    modelID: modelId),
                localPath: config.mtpDrafterPath,
                modelDirectory: modelDirectory,
                inlineDeclaration: inlineDeclaration,
                // `darkbloom start --local` is coordinator-independent and
                // never auto-downloads an assistant.
                allowDownload: false,
                environment: ProcessInfo.processInfo.environment))
    }
}
