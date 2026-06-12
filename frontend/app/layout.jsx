import "./globals.css";

export const metadata = {
  title: "Medical Reference Agent",
  description:
    "Agentic RAG over published medical textbooks & clinical references (educational, not medical advice)",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
