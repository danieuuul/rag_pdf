import { OllamaEmbeddings } from '@langchain/community/embeddings/ollama'

export function getEmbeddings() {
  const embeddings = new OllamaEmbeddings({
    model: 'nomic-embed-text', // default value
    baseUrl: 'http://localhost:11434', // default value
  })
  return embeddings
}
