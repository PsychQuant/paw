import Foundation
import Vapor

struct CompletionUsage: Content {
    let promptTokens: Int
    let completionTokens: Int
    let totalTokens: Int

    enum CodingKeys: String, CodingKey {
        case promptTokens = "prompt_tokens"
        case completionTokens = "completion_tokens"
        case totalTokens = "total_tokens"
    }
}

struct CompletionChoice: Content {
    let text: String
    let index: Int
    let logprobs: [String: Double]?
    let finishReason: String?

    init(text: String, index: Int = 0, logprobs: [String: Double]? = nil, finishReason: String?) {
        self.text = text
        self.index = index
        self.logprobs = logprobs
        self.finishReason = finishReason
    }

    enum CodingKeys: String, CodingKey {
        case text, index, logprobs
        case finishReason = "finish_reason"
    }
}

struct CompletionResponse: AsyncResponseEncodable, Content {
    let id: String
    let object: String
    let created: Int
    let model: String
    let choices: [CompletionChoice]
    let usage: CompletionUsage

    init(
        id: String = "cmpl-\(UUID().uuidString)", object: String = "text_completion",
        model: String, choices: [CompletionChoice], usage: CompletionUsage
    ) {
        self.id = id
        self.object = object
        self.created = Int(Date().timeIntervalSince1970)
        self.model = model
        self.choices = choices
        self.usage = usage
    }

    func encodeResponse(for request: Request) async throws -> Response {
        let response = Response(status: .ok)
        let encoder = JSONEncoder()
        encoder.keyEncodingStrategy = .convertToSnakeCase
        response.body = try .init(data: encoder.encode(self))
        response.headers.contentType = .json
        return response
    }

    enum CodingKeys: String, CodingKey {
        case id, object, created, model, choices, usage
    }
}

struct CompletionChunkResponse: Content {
    let id: String
    let object: String = "text_completion"
    let created: Int
    let choices: [Choice]
    let model: String
    let systemFingerprint: String

    struct Choice: Content {
        let text: String
        let index: Int = 0
        var logprobs: String?
        var finishReason: String?

        enum CodingKeys: String, CodingKey {
            case text, index, logprobs
            case finishReason = "finish_reason"
        }
    }

    init(
        completionId: String, requestedModel: String, nextChunk: String,
        systemFingerprint: String = "fp_\(UUID().uuidString)"
    ) {
        self.id = completionId
        self.created = Int(Date().timeIntervalSince1970)
        self.choices = [Choice(text: nextChunk)]
        self.model = requestedModel
        self.systemFingerprint = systemFingerprint
    }

    enum CodingKeys: String, CodingKey {
        case id, object, created, model, choices
        case systemFingerprint = "system_fingerprint"
    }
}

struct ChatMessageResponseData: Content {
    let role: String
    let content: String?
    let refusal: String? = nil

    enum CodingKeys: String, CodingKey {
        case role, content, refusal
    }

    init(role: String, content: String?) {
        self.role = role
        self.content = content
    }
}

struct ChatCompletionDelta: Content {
    var role: String?
    var content: String?
}

struct ChatCompletionChoiceDelta: Content {
    let index: Int
    let delta: ChatCompletionDelta
    let logprobs: String? = nil
    let finishReason: String?

    enum CodingKeys: String, CodingKey {
        case index, delta, logprobs
        case finishReason = "finish_reason"
    }
}

struct ChatCompletionChunkResponse: Content {
    let id: String
    let object: String = "chat.completion.chunk"
    let created: Int
    let model: String
    let systemFingerprint: String?
    let choices: [ChatCompletionChoiceDelta]

    init(
        id: String, created: Int = Int(Date().timeIntervalSince1970), model: String,
        systemFingerprint: String? = nil, choices: [ChatCompletionChoiceDelta]
    ) {
        self.id = id
        self.created = created
        self.model = model
        self.systemFingerprint = systemFingerprint
        self.choices = choices
    }

    enum CodingKeys: String, CodingKey {
        case id, object, created, model, choices
        case systemFingerprint = "system_fingerprint"
    }
}

struct ChatCompletionChoice: Content {
    let index: Int
    let message: ChatMessageResponseData
    let logprobs: String? = nil
    let finishReason: String

    enum CodingKeys: String, CodingKey {
        case index, message, logprobs
        case finishReason = "finish_reason"
    }
}

struct ChatCompletionResponse: AsyncResponseEncodable, Content {
    let id: String
    let object: String = "chat.completion"
    let created: Int
    let model: String
    let choices: [ChatCompletionChoice]
    let usage: CompletionUsage
    let systemFingerprint: String? = nil
    var serviceTier: String? = "default"

    init(
        id: String = "chatcmpl-\(UUID().uuidString)",
        created: Int = Int(Date().timeIntervalSince1970), model: String,
        choices: [ChatCompletionChoice], usage: CompletionUsage, serviceTier: String? = "default"
    ) {
        self.id = id
        self.created = created
        self.model = model
        self.choices = choices
        self.usage = usage
        self.serviceTier = serviceTier
    }

    func encodeResponse(for request: Request) async throws -> Response {
        let response = Response(status: .ok)
        let encoder = JSONEncoder()
        encoder.keyEncodingStrategy = .convertToSnakeCase
        response.body = try .init(data: encoder.encode(self))
        response.headers.contentType = .json
        return response
    }

    enum CodingKeys: String, CodingKey {
        case id, object, created, model, choices, usage
        case systemFingerprint = "system_fingerprint"
        case serviceTier = "service_tier"
    }
}

// MARK: - Model Info Types

struct ModelInfo: Content {
    let id: String
    var object: String = "model"
    let created = Int(Date().timeIntervalSince1970)
    let ownedBy: String = "user"

    enum CodingKeys: String, CodingKey {
        case id, object, created
        case ownedBy = "owned_by"
    }
}

struct ModelListResponse: Content {
    var object: String = "list"
    let data: [ModelInfo]
}

// MARK: - Cache Management Types

struct CacheStatusResponse: Content {
    let enabled: Bool
    let entryCount: Int
    let currentSizeMB: Double
    let maxSizeMB: Int
    let ttlMinutes: Int
    let stats: CacheStatsResponse
}

struct CacheStatsResponse: Content {
    let hits: Int
    let misses: Int
    let evictions: Int
    let hitRate: Double
    let totalTokensReused: Int
    let totalTokensProcessed: Int
    let averageTokensReused: Double
}

struct CacheClearResponse: Content {
    let success: Bool
    let message: String
}

// MARK: - Health Check Types

struct HealthResponse: Content {
    let status: String
    let version: String
}

// MARK: - Embedding Types

struct EmbeddingRequest: Content {
    let input: EmbeddingInput
    let model: String?
    let encoding_format: String?
    let dimensions: Int?
    let user: String?
    let batch_size: Int?
}

enum EmbeddingInput: Codable {
    case string(String)
    case array([String])

    init(from decoder: Swift.Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let str = try? container.decode(String.self) {
            self = .string(str)
        } else if let arr = try? container.decode([String].self) {
            self = .array(arr)
        } else {
            throw DecodingError.typeMismatch(
                EmbeddingInput.self,
                DecodingError.Context(
                    codingPath: decoder.codingPath,
                    debugDescription: "Expected String or [String]"))
        }
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .string(let str):
            try container.encode(str)
        case .array(let arr):
            try container.encode(arr)
        }
    }

    var values: [String] {
        switch self {
        case .string(let str): return [str]
        case .array(let arr): return arr
        }
    }
}

struct EmbeddingData: Content {
    var object: String = "embedding"
    let embedding: EmbeddingOutput
    let index: Int
}

enum EmbeddingOutput: Codable {
    case floats([Float])
    case base64(String)

    init(from decoder: Swift.Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let arr = try? container.decode([Float].self) {
            self = .floats(arr)
        } else if let str = try? container.decode(String.self) {
            self = .base64(str)
        } else {
            throw DecodingError.typeMismatch(
                EmbeddingOutput.self,
                DecodingError.Context(
                    codingPath: decoder.codingPath,
                    debugDescription: "Expected [Float] or String"))
        }
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .floats(let arr):
            try container.encode(arr)
        case .base64(let str):
            try container.encode(str)
        }
    }
}

struct UsageData: Content {
    let prompt_tokens: Int
    let total_tokens: Int
}

struct EmbeddingResponse: Content {
    var object: String = "list"
    let data: [EmbeddingData]
    let model: String
    let usage: UsageData
}
