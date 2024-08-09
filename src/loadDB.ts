import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter'
import { PDFLoader } from '@langchain/community/document_loaders/fs/pdf'
import { DirectoryLoader } from 'langchain/document_loaders/fs/directory'
import { Document } from 'langchain/document'
import { Chroma } from '@langchain/community/vectorstores/chroma'
import { getEmbeddings } from './getEmbeddings'

async function loadDocuments() {
  const directoryLoader = new DirectoryLoader('./data', {
    '.pdf': (path: string) => new PDFLoader(path)
  })
  return directoryLoader.load()
}

async function splitDocuments(documents: Document[]): Promise<Document[]> {
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 800,
    chunkOverlap: 80,
    lengthFunction: (text: string) => text.length,
    keepSeparator: false
  })
  return splitter.splitDocuments(documents)
}

function calculateChunkId(chunks: Document[]) {
  //This will create IDs like "data/monopoly.pdf:6:2"
  // Page Source : Page Number : Chunk Index

  let chunkId: string = ''
  let lastPageId: string = ''
  let currentChunkIndex: number = 0

  chunks.map((chunk) => {
    const source = chunk['metadata']['source']
    const page = chunk['metadata']['loc']['pageNumber']

    const currentPageId: string = `${source}:${page}`

    if (currentPageId !== lastPageId) {
      currentChunkIndex = 0
    } else {
      currentChunkIndex += 1
    }

    chunkId = `${source}:${page}:${currentChunkIndex}`
    lastPageId = currentPageId

    chunk['id'] = chunkId
  })

  return chunks
}
async function addToChroma(chunks: Document[]) {
  const vectorStore = new Chroma(getEmbeddings(), {
    collectionName: 'a-test-collection',
    url: 'http://localhost:8000', // Optional, will default to this value
    collectionMetadata: {
      'hnsw:space': 'cosine'
    }
  })

  const chunksWithIds = calculateChunkId(chunks)

  await vectorStore.addDocuments(chunks)
}

async function main() {
  const documents = await loadDocuments()
  const chunks = await splitDocuments(documents)
  await addToChroma(chunks)
}

main()
