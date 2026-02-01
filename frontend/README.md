# NeurIPS 2025 Paper Assistant - Frontend

A Next.js frontend for the CRAG (Corrective RAG) chatbot that helps explore NeurIPS 2025 papers.

## Features

- **Streaming Responses**: Real-time token streaming with SSE
- **Latency Display**: Shows pipeline step progress (e.g., "156ms - Reranking")
- **Structured Citations**: Clickable `[1]`, `[2]` references with expandable source cards
- **Session Persistence**: Maintains conversation history across page refreshes
- **Responsive Design**: Mobile-friendly chat interface

## Tech Stack

- **Framework**: Next.js 14 (App Router)
- **Styling**: Tailwind CSS + shadcn/ui
- **Language**: TypeScript
- **Deployment**: Vercel

## Getting Started

### Prerequisites

- Node.js 18+
- Running backend API (see parent directory)

### Installation

```bash
# Install dependencies
npm install

# Copy environment file
cp .env.example .env.local

# Edit .env.local to point to your backend
# NEXT_PUBLIC_API_URL=http://localhost:8000
```

### Development

```bash
# Start development server
npm run dev

# Open http://localhost:3000
```

### Production Build

```bash
npm run build
npm start
```

## Deployment to Vercel

1. Push your code to GitHub
2. Connect the repository to Vercel
3. Set environment variables:
   - `NEXT_PUBLIC_API_URL`: Your Railway backend URL
4. Deploy

## Project Structure

```
src/
├── app/
│   ├── layout.tsx       # Root layout
│   ├── page.tsx         # Main chat page
│   └── globals.css      # Global styles
├── components/
│   ├── chat/
│   │   ├── ChatContainer.tsx    # Main orchestrator
│   │   ├── MessageList.tsx      # Message display
│   │   ├── MessageBubble.tsx    # Individual messages
│   │   ├── ChatInput.tsx        # Input field
│   │   ├── StreamingStatus.tsx  # Pipeline status
│   │   └── CitationCard.tsx     # Citation display
│   └── ui/              # shadcn components
├── hooks/
│   └── useChat.ts       # SSE streaming hook
├── lib/
│   ├── api.ts           # API client
│   └── utils.ts         # Utilities
└── types/
    └── chat.ts          # TypeScript types
```

## Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `NEXT_PUBLIC_API_URL` | Backend API URL | `https://your-app.up.railway.app` |

## CORS Configuration

Make sure your backend allows requests from your Vercel domain. Update the CORS configuration in `crag/api.py` to include your production URL.
