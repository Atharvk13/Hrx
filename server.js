const express = require('express');
const axios = require('axios');
const pdf = require('pdf-parse');
const mammoth = require('mammoth');
const fs = require('fs');
const path = require('path');
const { RecursiveCharacterTextSplitter } = require('langchain/text_splitter');
 
const app = express();
app.use(express.json());
 
// Configuration
const CONFIG = {
    OPENAI_API_URL: 'https://dev-api.healthrx.co.in/sp-gw/api/openai/v1/chat/completions',
    BEARER_TOKEN: 'sk-spgw-api01-e3c8a211cb51528d1cd372d4ec7047a8',
    CHUNK_SIZE: 1000,
    CHUNK_OVERLAP: 200,
    MAX_CONTEXT_LENGTH: 8000,
    USE_LLM_EMBEDDINGS: false, // Set to true to use LLM for embeddings, false for local method
    SIMILARITY_THRESHOLD: 0.1, // Minimum similarity for considering a chunk relevant
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
        try {
            const response = await axios.get(url, {
                responseType: 'arraybuffer',
                timeout: 30000
            });
            return Buffer.from(response.data);
        } catch (error) {
            throw new Error(`Failed to download document: ${error.message}`);
        }
    }
 
    async extractTextFromPDF(buffer) {
        try {
            const data = await pdf(buffer);
            return data.text;
        } catch (error) {
            throw new Error(`PDF parsing failed: ${error.message}`);
        }
    }
 
    async extractTextFromDOCX(buffer) {
        try {
            const result = await mammoth.extractRawText({ buffer });
            return result.value;
        } catch (error) {
            throw new Error(`DOCX parsing failed: ${error.message}`);
        }
    }
 
    async processDocument(url) {
        const buffer = await this.downloadDocument(url);
        let text = '';
 
        // Determine file type from URL
        const urlLower = url.toLowerCase();
        
        if (urlLower.includes('.pdf') || urlLower.includes('pdf')) {
            text = await this.extractTextFromPDF(buffer);
        } else if (urlLower.includes('.docx') || urlLower.includes('docx')) {
            text = await this.extractTextFromDOCX(buffer);
        } else {
            // Try PDF first, then DOCX
            try {
                text = await this.extractTextFromPDF(buffer);
            } catch (pdfError) {
                try {
                    text = await this.extractTextFromDOCX(buffer);
                } catch (docxError) {
                    throw new Error('Unable to parse document as PDF or DOCX');
                }
            }
        }
 
        // Split text into chunks
        const chunks = await this.textSplitter.splitText(text);
        return { fullText: text, chunks };
    }
}
 
class EmbeddingService {
    constructor() {
        this.embeddings = new Map();
    }
 
    async generateEmbedding(text) {
        // Check configuration to decide embedding method
        if (CONFIG.USE_LLM_EMBEDDINGS) {
            try {
                const response = await this.callLLMForEmbedding(text);
                return this.convertResponseToEmbedding(response);
            } catch (error) {
                console.warn('LLM embedding generation failed, using enhanced local method');
                return this.createEnhancedEmbedding(text);
            }
        } else {
            // Use enhanced local embedding method directly
            return this.createEnhancedEmbedding(text);
        }
    }
 
    async callLLMForEmbedding(text) {
        try {
            const messages = [
                {
                    role: "system",
                    content: "Extract the 20 most important keywords and concepts from the following text. Return only a comma-separated list of keywords, no explanations."
                },
                {
                    role: "user",
                    content: text.substring(0, 2000) // Limit text length for efficiency
                }
            ];
 
            const response = await axios.post(CONFIG.OPENAI_API_URL, {
                messages: messages,
                model: "gpt-4o",
                temperature: 0.1,
                max_tokens: 200
            }, {
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${CONFIG.BEARER_TOKEN}`
                },
                timeout: 30000
            });
 
            return response.data.choices[0].message.content;
        } catch (error) {
            throw new Error(`LLM embedding call failed: ${error.message}`);
        }
    }
 
    convertResponseToEmbedding(keywordString) {
        const keywords = keywordString.toLowerCase().split(',').map(k => k.trim());
        const embedding = new Array(384).fill(0);
        
        // Create embedding based on keywords and their semantic relationships
        keywords.forEach((keyword, idx) => {
            const positions = this.getSemanticPositions(keyword);
            positions.forEach(pos => {
                embedding[pos % 384] += (keywords.length - idx) / keywords.length; // Weight by importance
            });
        });
        
        // Normalize the embedding
        const magnitude = Math.sqrt(embedding.reduce((sum, val) => sum + val * val, 0));
        return magnitude > 0 ? embedding.map(val => val / magnitude) : embedding;
    }
 
    getSemanticPositions(keyword) {
        // Generate multiple positions for each keyword to capture semantic relationships
        const positions = [];
        const baseHash = this.simpleHash(keyword);
        
        // Add base position
        positions.push(baseHash);
        
        // Add semantic variations
        const chars = keyword.split('');
        for (let i = 0; i < Math.min(chars.length, 3); i++) {
            positions.push(this.simpleHash(chars.slice(i).join('')));
        }
        
        // Add length-based position
        positions.push(this.simpleHash(keyword.length.toString()));
        
        return positions;
    }
 
    createEnhancedEmbedding(text) {
        // Enhanced embedding creation using multiple techniques
        const words = text.toLowerCase().split(/\s+/).filter(word => word.length > 2);
        const embedding = new Array(384).fill(0);
        
        // 1. Term frequency approach
        const wordCount = {};
        words.forEach(word => {
            wordCount[word] = (wordCount[word] || 0) + 1;
        });
        
        // 2. Position-based weighting (earlier words get higher weight)
        Object.entries(wordCount).forEach(([word, count]) => {
            const positions = this.getSemanticPositions(word);
            const tf = count / words.length;
            const positionWeight = 1 / (words.indexOf(word) + 1); // Earlier = higher weight
            
            positions.forEach(pos => {
                embedding[pos % 384] += tf * positionWeight;
            });
        });
        
        // 3. Add n-gram features for better context
        this.addNgramFeatures(words, embedding);
        
        // 4. Add document structure features
        this.addStructureFeatures(text, embedding);
        
        // Normalize the embedding
        const magnitude = Math.sqrt(embedding.reduce((sum, val) => sum + val * val, 0));
        return magnitude > 0 ? embedding.map(val => val / magnitude) : embedding;
    }
    
    addNgramFeatures(words, embedding) {
        // Add bigram features
        for (let i = 0; i < words.length - 1; i++) {
            const bigram = words[i] + '_' + words[i + 1];
            const hash = this.simpleHash(bigram);
            embedding[hash % 384] += 0.5; // Lower weight for n-grams
        }
        
        // Add trigram features for key phrases
        for (let i = 0; i < words.length - 2; i++) {
            const trigram = words[i] + '_' + words[i + 1] + '_' + words[i + 2];
            const hash = this.simpleHash(trigram);
            embedding[hash % 384] += 0.3;
        }
    }
    
    addStructureFeatures(text, embedding) {
        // Add features based on document structure
        const sentences = text.split(/[.!?]+/).length;
        const avgSentenceLength = text.length / sentences;
        const hasNumbers = /\d/.test(text);
        const hasDates = /\b\d{1,2}[-\/]\d{1,2}[-\/]\d{2,4}\b/.test(text);
        const hasPercentages = /%/.test(text);
        const hasCurrency = /\$|₹|€|£/.test(text);
        
        // Encode structural features
        embedding[380] += avgSentenceLength / 100; // Normalized avg sentence length
        embedding[381] += hasNumbers ? 1 : 0;
        embedding[382] += hasDates ? 1 : 0;
        embedding[383] += (hasPercentages ? 0.5 : 0) + (hasCurrency ? 0.5 : 0);
    }
 
    simpleHash(str) {
        let hash = 0;
        for (let i = 0; i < str.length; i++) {
            const char = str.charCodeAt(i);
            hash = ((hash << 5) - hash) + char;
            hash = hash & hash;
        }
        return Math.abs(hash);
    }
 
    cosineSimilarity(vecA, vecB) {
        const dotProduct = vecA.reduce((sum, a, i) => sum + a * vecB[i], 0);
        const magnitudeA = Math.sqrt(vecA.reduce((sum, a) => sum + a * a, 0));
        const magnitudeB = Math.sqrt(vecB.reduce((sum, b) => sum + b * b, 0));
        
        if (magnitudeA === 0 || magnitudeB === 0) return 0;
        return dotProduct / (magnitudeA * magnitudeB);
    }
 
    async indexChunks(chunks) {
        const indexedChunks = [];
        
        for (let i = 0; i < chunks.length; i++) {
            const chunk = chunks[i];
            const embedding = await this.generateEmbedding(chunk);
            indexedChunks.push({
                id: i,
                text: chunk,
                embedding: embedding,
                metadata: {
                    length: chunk.length,
                    position: i
                }
            });
        }
        
        return indexedChunks;
    }
 
    async searchSimilarChunks(query, indexedChunks, topK = 5) {
        const queryEmbedding = await this.generateEmbedding(query);
        
        // Calculate similarities with enhanced scoring
        const similarities = indexedChunks.map(chunk => {
            const semanticSimilarity = this.cosineSimilarity(queryEmbedding, chunk.embedding);
            
            // Add keyword-based boost for direct matches
            const keywordBoost = this.calculateKeywordBoost(query, chunk.text);
            
            // Add position-based scoring (earlier chunks might be more important)
            const positionScore = 1 / (chunk.metadata.position + 1) * 0.1;
            
            const combinedScore = semanticSimilarity + keywordBoost + positionScore;
            
            return {
                ...chunk,
                similarity: combinedScore,
                semanticScore: semanticSimilarity,
                keywordScore: keywordBoost,
                positionScore: positionScore
            };
        });
 
        return similarities
            .sort((a, b) => b.similarity - a.similarity)
            .slice(0, topK);
    }
    
    calculateKeywordBoost(query, text) {
        const queryWords = query.toLowerCase().split(/\s+/).filter(word => word.length > 2);
        const textLower = text.toLowerCase();
        
        let boost = 0;
        let totalMatches = 0;
        
        queryWords.forEach(word => {
            // Exact word match
            const exactMatches = (textLower.match(new RegExp(`\\b${word}\\b`, 'g')) || []).length;
            totalMatches += exactMatches;
            boost += exactMatches * 0.1;
            
            // Partial word match
            if (textLower.includes(word)) {
                boost += 0.05;
            }
        });
        
        // Normalize boost by query length
        return Math.min(boost / queryWords.length, 0.3); // Cap at 0.3
    }
}
 
class QueryProcessor {
    constructor() {
        this.documentProcessor = new DocumentProcessor();
        this.embeddingService = new EmbeddingService();
    }
 
    async callLLM(messages, temperature = 0.1) {
        try {
            const response = await axios.post(CONFIG.OPENAI_API_URL, {
                messages: messages,
                model: "gpt-4o",
                temperature: temperature,
                max_tokens: 1000
            }, {
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${CONFIG.BEARER_TOKEN}`
                },
                timeout: 60000
            });
 
            return response.data.choices[0].message.content;
        } catch (error) {
            console.error('LLM API Error:', error.response?.data || error.message);
            throw new Error(`LLM API failed: ${error.message}`);
        }
    }
 
    async answerQuestion(question, relevantChunks, fullContext) {
        const contextText = relevantChunks
            .map(chunk => `[Relevance: ${(chunk.similarity * 100).toFixed(1)}%]\n${chunk.text}`)
            .join('\n\n---\n\n');
 
        const messages = [
            {
                role: "system",
                content: `You are an expert document analyst specializing in insurance, legal, HR, and compliance domains.
                
Your task is to answer questions based STRICTLY on the provided document context.
 
Guidelines:
1. Only use information explicitly stated in the provided context
2. If information is not found in the context, clearly state this
3. Be precise and specific in your answers
4. Quote relevant sections when appropriate
5. Provide clear, structured responses
6. For insurance/policy questions, focus on coverage details, conditions, waiting periods, and exclusions
7. Always maintain accuracy over completeness
 
Context relevance scores are provided to help you prioritize information.`
            },
            {
                role: "user",
                content: `Document Context:\n${contextText}\n\nQuestion: ${question}\n\nPlease provide a comprehensive answer based solely on the provided context.`
            }
        ];
 
        const answer = await this.callLLM(messages);
        
        return {
            answer: answer.trim(),
            relevantSections: relevantChunks.map(chunk => ({
                text: chunk.text.substring(0, 200) + '...',
                similarity: Math.round(chunk.similarity * 100),
                position: chunk.metadata.position
            })),
            confidence: this.calculateConfidence(relevantChunks),
            reasoning: this.generateReasoning(question, relevantChunks)
        };
    }
 
    calculateConfidence(relevantChunks) {
        if (relevantChunks.length === 0) return 0;
        
        const avgSimilarity = relevantChunks.reduce((sum, chunk) => sum + chunk.similarity, 0) / relevantChunks.length;
        const topSimilarity = relevantChunks[0]?.similarity || 0;
        
        // Confidence based on similarity scores and number of relevant chunks
        const confidence = (avgSimilarity * 0.6 + topSimilarity * 0.4) * Math.min(relevantChunks.length / 3, 1);
        
        return Math.round(confidence * 100);
    }
 
    generateReasoning(question, relevantChunks) {
        const topChunk = relevantChunks[0];
        if (!topChunk) {
            return "No relevant information found in the document.";
        }
 
        return `Answer derived from ${relevantChunks.length} relevant document section(s). ` +
               `Primary source has ${topChunk.similarity > 0.7 ? 'high' : topChunk.similarity > 0.4 ? 'moderate' : 'low'} relevance (${Math.round(topChunk.similarity * 100)}%). ` +
               `Information extracted from document position ${topChunk.metadata.position + 1}.`;
    }
 
    async processQueries(documentUrl, questions) {
        console.log('Processing document:', documentUrl);
        
        // Extract and process document
        const { fullText, chunks } = await this.documentProcessor.processDocument(documentUrl);
        console.log(`Document processed: ${chunks.length} chunks created`);
        
        // Create embeddings index
        const indexedChunks = await this.embeddingService.indexChunks(chunks);
        console.log('Document indexed with embeddings');
        
        const results = [];
        
        for (let i = 0; i < questions.length; i++) {
            const question = questions[i];
            console.log(`Processing question ${i + 1}/${questions.length}: ${question.substring(0, 50)}...`);
            
            try {
                // Find relevant chunks
                const relevantChunks = await this.embeddingService.searchSimilarChunks(
                    question,
                    indexedChunks,
                    5
                );
                
                // Generate answer
                const result = await this.answerQuestion(question, relevantChunks, fullText);
                results.push(result.answer);
                
            } catch (error) {
                console.error(`Error processing question ${i + 1}:`, error.message);
                results.push(`Error processing question: ${error.message}`);
            }
        }
        
        return results;
    }
}
 
// API Routes
const queryProcessor = new QueryProcessor();
 
app.post('/hackrx/run', async (req, res) => {
    try {
        const { documents, questions } = req.body;
        
        // Validation
        if (!documents || !questions || !Array.isArray(questions)) {
            return res.status(400).json({
                error: 'Invalid request format. Expected documents URL and questions array.'
            });
        }
        
        console.log(`Processing ${questions.length} questions for document: ${documents}`);
        
        const answers = await queryProcessor.processQueries(documents, questions);
        
        res.json({ answers });
        
    } catch (error) {
        console.error('Processing error:', error);
        res.status(500).json({
            error: 'Failed to process request',
            details: error.message
        });
    }
});
 
// Health check endpoint
app.get('/health', (req, res) => {
    res.json({
        status: 'healthy',
        timestamp: new Date().toISOString(),
        version: '1.0.0'
    });
});
 
// Enhanced endpoint with detailed response
app.post('/hackrx/run/detailed', async (req, res) => {
    try {
        const { documents, questions } = req.body;
        
        if (!documents || !questions || !Array.isArray(questions)) {
            return res.status(400).json({
                error: 'Invalid request format'
            });
        }
        
        const { fullText, chunks } = await queryProcessor.documentProcessor.processDocument(documents);
        const indexedChunks = await queryProcessor.embeddingService.indexChunks(chunks);
        
        const detailedResults = [];
        
        for (const question of questions) {
            const relevantChunks = await queryProcessor.embeddingService.searchSimilarChunks(
                question,
                indexedChunks,
                5
            );
            
            const result = await queryProcessor.answerQuestion(question, relevantChunks, fullText);
            detailedResults.push({
                question,
                answer: result.answer,
                confidence: result.confidence,
                reasoning: result.reasoning,
                relevantSections: result.relevantSections
            });
        }
        
        res.json({
            document_info: {
                url: documents,
                total_chunks: chunks.length,
                processed_at: new Date().toISOString()
            },
            results: detailedResults
        });
        
    } catch (error) {
        console.error('Detailed processing error:', error);
        res.status(500).json({
            error: 'Failed to process request',
            details: error.message
        });
    }
});
 
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
    console.log(`🚀 Intelligent Document Query System running on port ${PORT}`);
    console.log(`📋 API Endpoints:`);
    console.log(`   POST /hackrx/run - Standard query processing`);
    console.log(`   POST /hackrx/run/detailed - Detailed query processing with metadata`);
    console.log(`   GET  /health - Health check`);
});
 
module.exports = app;
 
