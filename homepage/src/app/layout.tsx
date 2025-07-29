import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "MyHomepage - Modern Next.js Website",
  description: "A modern, responsive website built with Next.js, React, and TypeScript. Deployed on Vercel with serverless architecture.",
  keywords: ["Next.js", "React", "TypeScript", "Tailwind CSS", "Vercel", "Serverless"],
  authors: [{ name: "MyHomepage" }],
  creator: "MyHomepage",
  publisher: "MyHomepage",
  formatDetection: {
    email: false,
    address: false,
    telephone: false,
  },
  metadataBase: new URL('https://myhomepage.vercel.app'),
  alternates: {
    canonical: '/',
  },
  openGraph: {
    title: "MyHomepage - Modern Next.js Website",
    description: "A modern, responsive website built with Next.js, React, and TypeScript.",
    url: 'https://myhomepage.vercel.app',
    siteName: 'MyHomepage',
    images: [
      {
        url: '/og-image.jpg',
        width: 1200,
        height: 630,
        alt: 'MyHomepage',
      },
    ],
    locale: 'en_US',
    type: 'website',
  },
  twitter: {
    card: 'summary_large_image',
    title: "MyHomepage - Modern Next.js Website",
    description: "A modern, responsive website built with Next.js, React, and TypeScript.",
    images: ['/og-image.jpg'],
  },
  robots: {
    index: true,
    follow: true,
    googleBot: {
      index: true,
      follow: true,
      'max-video-preview': -1,
      'max-image-preview': 'large',
      'max-snippet': -1,
    },
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="scroll-smooth">
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased`}
      >
        {children}
      </body>
    </html>
  );
}
