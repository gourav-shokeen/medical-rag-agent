import "./globals.css";

export const metadata = {
  title: "10-K Analyst Agent",
  description: "Agentic RAG over SEC 10-K filings (AAPL, MSFT, NVDA)",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
