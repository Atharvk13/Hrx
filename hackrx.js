const axios = require('axios');
const pdfParse = require('pdf-parse');

module.exports = async (req, res) => {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Only POST allowed' });
  }

  try {
    const { documents, questions } = req.body;

    if (!documents || !questions || !Array.isArray(questions)) {
      return res.status(400).json({ error: 'Invalid request format.' });
    }

    // Download and extract PDF text
    const response = await axios.get(documents, { responseType: 'arraybuffer' });
    const data = await pdfParse(response.data);
    const context = data.text;

    // Ask OpenAI
    const answers = await Promise.all(
      questions.map(async (q) => {
        const payload = {
          model: 'gpt-4o',
          messages: [
            { role: 'system', content: 'You are a helpful assistant. Answer based strictly on the given document.' },
            { role: 'user', content: `Context:\n${context}\n\nQuestion: ${q}` }
          ]
        };

        const aiRes = await axios.post(
          process.env.OPENAI_BASE_URL || 'https://api.openai.com/v1/chat/completions',
          payload,
          {
            headers: {
              'Content-Type': 'application/json',
              Authorization: `Bearer ${process.env.OPENAI_API_KEY}`
            }
          }
        );

        return {
          question: q,
          answer: aiRes.data.choices[0].message.content.trim()
        };
      })
    );

    res.status(200).json({ status: 'success', answers });

  } catch (error) {
    console.error('Error:', error.message);
    res.status(500).json({ error: 'Something went wrong.' });
  }
};
