const express = require('express');
const axios = require('axios');
const pdf = require('pdf-parse');
const mammoth = require('mammoth');
const { RecursiveCharacterTextSplitter } = require('langchain/text_splitter');

const app = express();
app.use(express.json());

const CONFIG = {
    OPENAI_API_URL: 'https://dev-api.healthrx.co.in/sp-gw/api/openai/v1/chat/completions',
    BEARER_TOKEN: 'sk-spgw-api01-e3c8a211cb51528d1cd372d4ec7047a8',
    CHUNK_SIZE: 1000,
    CHUNK_OVERLAP: 200,
    MAX_CONTEXT_LENGTH: 8000,
    USE_LLM_EMBEDDINGS: false,
    SIMILARITY_THRESHOLD: 0.1,
    MAX_RETRIES: 3
};

class DocumentProcessor {
    constructor() {
        this.textSplitter = new RecursiveCharacterTextSplitter({
            chunkSize: CONFIG.CHUNK_SIZE,
            chunkOverlap: CONFIG.CHUNK_OVERLAP
        });
    }

    async downloadDocument(url) {
        const response = await axios.get(url, { responseType: 'arraybuffer', timeout: 30000 });
        return Buffer.from(response.data);
    }

    async extractText(buffer, url) {
        if (url.toLowerCase().includes('.pdf')) {
            return (await pdf(buffer)).text;
        } else if (url.toLowerCase().includes('.docx')) {
            return (await mammoth.extractRawText({ buffer })).value;
        }

        try {
            return (await pdf(buffer)).text;
        } catch {
            return (await mammoth.extractRawText({ buffer })).value;
        }
    }

    async processDocument(url) {
        const buffer = await this.downloadDocument(url);
        const text = await this.extractText(buffer, url);
        const chunks = await this.textSplitter.splitText(text);
        return { fullText: text, chunks };
    }
}

class EmbeddingService {
    constructor() {
        this.embeddings = new Map();
    }

    async generateEmbedding(text) {
        return CONFIG.USE_LLM_EMBEDDINGS ? await this.callLLMEmbedding(text) : this.localEmbedding(text);
    }

    async callLLMEmbedding(text) {
        const messages = [
            { role: 'system', content: 'Extract top 20 keywords from the text.' },
            { role: 'user', content: text.substring(0, 2000) }
        ];
        const res = await axios.post(CONFIG.OPENAI_API_URL, {
            messages,
            model: 'gpt-4o',
            temperature: 0.1,
            max_tokens: 200
        }, {
            headers: { 'Authorization': `Bearer ${CONFIG.BEARER_TOKEN}` },
            timeout: 30000
        });
        return this.localEmbedding(res.data.choices[0].message.content);
    }

    localEmbedding(text) {
        const words = text.toLowerCase().split(/\W+/).filter(w => w.length > 2);
        const embedding = new Array(384).fill(0);
        const freq = words.reduce((acc, word) => (acc[word] = (acc[word] || 0) + 1, acc), {});

        Object.entries(freq).forEach(([word, count]) => {
            const hash = this.hash(word);
            embedding[hash % 384] += count;
        });

        const mag = Math.sqrt(embedding.reduce((sum, v) => sum + v * v, 0));
        return embedding.map(v => v / mag);
    }

    hash(str) {
        let hash = 0;
        for (let i = 0; i < str.length; i++) hash = ((hash << 5) - hash) + str.charCodeAt(i);
        return Math.abs(hash);
    }

    cosineSimilarity(a, b) {
        const dot = a.reduce((sum, val, i) => sum + val * b[i], 0);
        const magA = Math.sqrt(a.reduce((sum, v) => sum + v * v, 0));
        const magB = Math.sqrt(b.reduce((sum, v) => sum + v * v, 0));
        return magA && magB ? dot / (magA * magB) : 0;
    }

    async indexChunks(chunks) {
        return await Promise.all(chunks.map(async (text, i) => ({
            id: i,
            text,
            embedding: await this.generateEmbedding(text),
            metadata: { length: text.length, position: i }
        })));
    }

    async searchSimilarChunks(query, indexedChunks, topK = 5) {
        const queryEmbedding = await this.generateEmbedding(query);
        const scored = indexedChunks.map(chunk => ({
            ...chunk,
            similarity: this.cosineSimilarity(queryEmbedding, chunk.embedding)
        }));
        return scored.sort((a, b) => b.similarity - a.similarity).slice(0, topK);
    }
}

class QueryProcessor {
    constructor() {
        this.docProc = new DocumentProcessor();
        this.embedService = new EmbeddingService();
    }

    async callLLM(messages) {
        const res = await axios.post(CONFIG.OPENAI_API_URL, {
            messages,
            model: 'gpt-4o',
            temperature: 0.1,
            max_tokens: 1000
        }, {
            headers: { 'Authorization': `Bearer ${CONFIG.BEARER_TOKEN}` },
            timeout: 60000
        });
        return res.data.choices[0].message.content;
    }

    async answerQuestion(question, chunks, fullText) {
        const context = chunks.map(c => c.text).join('\n---\n');
        const messages = [
            { role: 'system', content: 'Answer strictly based on provided context.' },
            { role: 'user', content: `Context:\n${context}\n\nQuestion: ${question}` }
        ];
        const answer = await this.callLLM(messages);
        return {
            answer,
            contextSnippets: chunks.map(c => ({ text: c.text.slice(0, 200) + '...', score: c.similarity }))
        };
    }

    async processQueries(documentUrl, questions) {
        const { fullText, chunks } = await this.docProc.processDocument(documentUrl);
        const indexedChunks = await this.embedService.indexChunks(chunks);
        const results = [];

        for (const question of questions) {
            const relevant = await this.embedService.searchSimilarChunks(question, indexedChunks, 5);
            const result = await this.answerQuestion(question, relevant, fullText);
            results.push(result);
        }

        return results;
    }
}

const qp = new QueryProcessor();

app.post('/hackrx/run', async (req, res) => {
    try {
        const { documents, questions } = req.body;
        if (!documents || !Array.isArray(questions)) return res.status(400).json({ error: 'Invalid input' });
        const results = await qp.processQueries(documents, questions);
        res.json({ answers: results.map(r => r.answer) });
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

app.post('/hackrx/run/detailed', async (req, res) => {
    try {
        const { documents, questions } = req.body;
        if (!documents || !Array.isArray(questions)) return res.status(400).json({ error: 'Invalid input' });

        const { fullText, chunks } = await qp.docProc.processDocument(documents);
        const indexedChunks = await qp.embedService.indexChunks(chunks);
        const results = [];

        for (const question of questions) {
            const relevant = await qp.embedService.searchSimilarChunks(question, indexedChunks, 5);
            const result = await qp.answerQuestion(question, relevant, fullText);
            results.push({ question, ...result });
        }

        res.json({
            document: { url: documents, chunks: chunks.length, timestamp: new Date() },
            results
        });
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

app.get('/health', (req, res) => {
    res.json({ status: 'healthy', version: '1.0.0', timestamp: new Date() });
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
});

module.exports = app;
