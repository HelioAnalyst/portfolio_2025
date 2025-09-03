import { TempoInit } from "@/components/tempo-init";
import type { Metadata } from "next";
import Script from "next/script";
import "./globals.css";
import { ThemeProvider } from "@/components/theme-provider";
import { ThemeSwitcher } from "@/components/theme-switcher";
import Navigation from "@/components/Navigation";
import Footer from "@/components/Footer";
import icons from "@/public/favicon.svg";

export const metadata: Metadata = {
  title: "Helio | The Analyst - Python Developer & Data Analyst",
  description:
    "Mid-level Python Developer & Data Analyst specializing in API integration, automation, web scraping, and data processing solutions. Expert in Shopify API, Selenium, and process optimization.",
  icons: icons,
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <Script src="https://api.tempo.build/proxy-asset?url=https://storage.googleapis.com/tempo-public-assets/error-handling.js" />
      <body className="font-space-grotesk">
        <ThemeProvider
          attribute="class"
          defaultTheme="system"
          enableSystem
          disableTransitionOnChange
        >
          <div className="fixed right-4 top-20 z-50">
            <ThemeSwitcher />
          </div>
          <Navigation />
          <main>{children}</main>
          <Footer />
          <TempoInit />
        </ThemeProvider>
      </body>
    </html>
  );
}
