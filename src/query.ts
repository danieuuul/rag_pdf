import { PromptTemplate } from '@langchain/core/prompts'
import { getEmbeddings } from './getEmbeddings'
import { Chroma } from '@langchain/community/vectorstores/chroma'
import { Document } from 'langchain/document'
import { Ollama } from '@langchain/ollama'

const PROMPT_TEMPLATE = `Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}`

async function main() {
  const vectorStore = new Chroma(getEmbeddings(), {
    collectionName: 'pdf-collection',
    url: 'http://localhost:8000', // Optional, will default to this value
    collectionMetadata: {
      'hnsw:space': 'cosine',
    },
  })

  const model = new Ollama({
    model: 'llama3.1', // Default value
    temperature: 0,
    maxRetries: 2,
  })

  const question =
    'How much total money does a player start with in Monopoly? (Answer with the number only)?'

  const results = await vectorStore.similaritySearchWithScore(question, 5)

  const prompt = PromptTemplate.fromTemplate(PROMPT_TEMPLATE)

  const context = results
    .map(([doc, _score]: [Document, number]) => doc.pageContent)
    .reduce((acc, pageContent) => acc.concat(pageContent).concat('\n\n---\n\n'))

  const chain = prompt.pipe(model)
  const answer = await chain.invoke({
    context,
    question,
  })

  console.log(`
    Resposta: ${answer}
    
    Contexto: ${context});
    }
    `)
}

main()
