import { Github, Linkedin, Mail, Phone, MapPin } from "lucide-react";
import Link from "next/link";

export default function Footer() {
  const currentYear = new Date().getFullYear();

  const socialLinks = [
    {
      href: "https://github.com/HelioAnalyst",
      icon: Github,
      label: "GitHub",
    },
    {
      href: "https://www.linkedin.com/in/helio-bruno-garcia-massadico/",
      icon: Linkedin,
      label: "LinkedIn",
    },
    {
      href: "mailto:contact@heliotheanalyst.co.uk",
      icon: Mail,
      label: "Email",
    },
  ];

  const quickLinks = [
    { href: "/", label: "Home" },
    { href: "/cv", label: "CV" },
    { href: "/contact", label: "Contact" },
  ];

  const services = [
    "Python Development",
    "Data Analysis",
    "API Integration",
    "Web Scraping",
    "Process Automation",
    "Selenium Automation",
  ];

  return (
    <footer className="bg-primary text-primary-foreground">
      <div className="container mx-auto px-4 py-12">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
          {/* Brand Section */}
          <div className="lg:col-span-1">
            <div className="flex items-center gap-3 mb-4">
              <div className="logo-mark">
                <svg width="32" height="32" viewBox="0 0 120 120">
                  <circle
                    cx="60"
                    cy="60"
                    r="50"
                    fill="none"
                    stroke="white"
                    strokeWidth="6"
                    strokeDasharray="280 35"
                    strokeDashoffset="-17.5"
                    className="opacity-90"
                  />
                  <g fill="white">
                    <rect x="42" y="35" width="6" height="50" />
                    <rect x="72" y="35" width="6" height="50" />
                    <rect x="42" y="57" width="36" height="6" />
                  </g>
                </svg>
              </div>
              <div>
                <span className="brand-wordmark text-xl font-semibold">
                  HELIO
                </span>
                <div className="brand-tagline text-xs tracking-wider text-primary-foreground/70">
                  THE ANALYST
                </div>
              </div>
            </div>
            <p className="text-primary-foreground/70 text-sm leading-relaxed mb-4">
              Mid-level Python Developer with 5+ years of experience in API
              integration, automation, and data processing. Specialized in
              Shopify API integrations and web scraping with Selenium.
            </p>
            <div className="flex gap-4">
              {socialLinks.map((link) => {
                const Icon = link.icon;
                return (
                  <a
                    key={link.href}
                    href={link.href}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-primary-foreground/70 hover:text-primary-foreground transition-colors"
                    aria-label={link.label}
                  >
                    <Icon className="w-5 h-5" />
                  </a>
                );
              })}
            </div>
          </div>

          {/* Quick Links */}
          <div>
            <h3 className="font-semibold text-lg mb-4">Quick Links</h3>
            <ul className="space-y-2">
              {quickLinks.map((link) => (
                <li key={link.href}>
                  <Link
                    href={link.href}
                    className="text-primary-foreground/70 hover:text-primary-foreground transition-colors text-sm"
                  >
                    {link.label}
                  </Link>
                </li>
              ))}
            </ul>
          </div>

          {/* Services */}
          <div>
            <h3 className="font-semibold text-lg mb-4">Services</h3>
            <ul className="space-y-2">
              {services.map((service) => (
                <li
                  key={service}
                  className="text-primary-foreground/70 text-sm"
                >
                  {service}
                </li>
              ))}
            </ul>
          </div>

          {/* Contact Info */}
          <div>
            <h3 className="font-semibold text-lg mb-4">Contact</h3>
            <div className="space-y-3">
              <div className="flex items-center gap-3 text-primary-foreground/70 text-sm">
                <Mail className="w-4 h-4" />
                <span>contact@heliotheanalyst.co.uk</span>
              </div>
              <div className="flex items-center gap-3 text-primary-foreground/70 text-sm">
                <Phone className="w-4 h-4" />
                <span>+44 7818 351 949</span>
              </div>
              <div className="flex items-center gap-3 text-primary-foreground/70 text-sm">
                <MapPin className="w-4 h-4" />
                <span>Southampton, UK</span>
              </div>
            </div>
          </div>
        </div>

        {/* Bottom Section */}
        <div className="border-t border-primary-foreground/20 mt-8 pt-8 flex flex-col md:flex-row justify-between items-center">
          <p className="text-primary-foreground/70 text-sm">
            © {currentYear} Helio Bruno Garcia Massadico. All rights reserved.
          </p>
          <p className="text-gray-400 text-sm mt-2 md:mt-0">
            Built with Next.js & Tailwind CSS
          </p>
        </div>
      </div>
    </footer>
  );
}
