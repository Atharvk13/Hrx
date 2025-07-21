const express = require('express');
const axios = require('axios');
const pdfParse = require('pdf-parse');
const dotenv = require('dotenv');
 
dotenv.config();
 
const app = express();
app.use(express.json({ limit: '50mb' }));
 
// Helper: Download PDF and extract text
async function extractTextFromPDF(pdfUrl) {
  const response = await axios.get(pdfUrl, { responseType: 'arraybuffer' });
  const data = await pdfParse(response.data);
  return data.text;
}
 
// Helper: Query OpenAI
async function askOpenAI(question, context) {
  const payload = {
    model: 'gpt-4o',
    messages: [
      {
        role: 'system',
        content: 'You are a helpful assistant. Answer based strictly on the given document.'
      },
      {
        role: 'user',
        content: `Context:\n${context}\n\nQuestion: ${question}`
      }
    ]
  };
 
  const config = {
    method: 'post',
    url: process.env.OPENAI_BASE_URL,
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${process.env.OPENAI_API_KEY}`
    },
    data: JSON.stringify(payload)
  };
 
  const res = await axios(config);
  return res.data.choices[0].message.content.trim();
}
 
// API Route
app.post('/hackrx/run', async (req, res) => {
  try {
    const { documents, questions } = req.body;
 
    if (!documents || !questions || !Array.isArray(questions)) {
      return res.status(400).json({ error: 'Invalid request format.' });
    }
 
    const context = await extractTextFromPDF(documents);
 
    const answers = await Promise.all(
      questions.map(async (q) => {
        const answer = await askOpenAI(q, context);
        return { question: q, answer };
      })
    );
 
    res.json({ status: 'success', answers });
  } catch (error) {
    console.error('Error:', error.message);
    res.status(500).json({ error: 'Something went wrong.' });
  }
});
 
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
});
