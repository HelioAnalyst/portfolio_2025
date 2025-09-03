"use client";

import { useState } from "react";
import { Button } from "./ui/button";
import { Menu, X, Code, User, Mail, FileText } from "lucide-react";
import Link from "next/link";

export default function Navigation() {
  const [isMenuOpen, setIsMenuOpen] = useState(false);

  const toggleMenu = () => setIsMenuOpen(!isMenuOpen);

  const navItems = [
    { href: "/", label: "Home", icon: Code },
    { href: "/blog", label: "Blog", icon: FileText },
    { href: "/cv", label: "CV", icon: User },
    { href: "/contact", label: "Contact", icon: Mail },
  ];

  return (
    <nav className="bg-background/95 backdrop-blur-sm border-b border-border sticky top-0 z-50">
      <div className="container mx-auto px-4">
        <div className="flex items-center justify-between h-16">
          {/* Logo */}
          <Link
            href="/"
            className="flex items-center gap-3 hover:opacity-80 transition-opacity"
          >
            <div className="logo-mark">
              <svg width="32" height="32" viewBox="0 0 120 120">
                <circle
                  cx="60"
                  cy="60"
                  r="50"
                  fill="none"
                  stroke="hsl(var(--brand-plum))"
                  strokeWidth="6"
                  strokeDasharray="280 35"
                  strokeDashoffset="-17.5"
                  className="opacity-90"
                />
                <g fill="hsl(var(--brand-plum))">
                  <rect x="42" y="35" width="6" height="50" />
                  <rect x="72" y="35" width="6" height="50" />
                  <rect x="42" y="57" width="36" height="6" />
                </g>
              </svg>
            </div>
            <div>
              <span className="brand-wordmark text-xl font-semibold text-brand-plum dark:text-brand-aqua">
                HELIO
              </span>
              <div className="brand-tagline text-xs tracking-wider text-gray-600 dark:text-foreground">
                THE ANALYST
              </div>
            </div>
          </Link>

          {/* Desktop Navigation */}
          <div className="hidden md:flex items-center gap-8">
            {navItems.map((item) => {
              const Icon = item.icon;
              return (
                <Link
                  key={item.href}
                  href={item.href}
                  className="flex items-center gap-2 text-foreground/80 hover:text-primary transition-colors font-medium"
                >
                  <Icon className="w-4 h-4" />
                  {item.label}
                </Link>
              );
            })}
          </div>

          {/* Mobile Menu Button */}
          <Button
            variant="ghost"
            size="sm"
            className="md:hidden"
            onClick={toggleMenu}
            aria-label="Toggle menu"
          >
            {isMenuOpen ? (
              <X className="w-5 h-5" />
            ) : (
              <Menu className="w-5 h-5" />
            )}
          </Button>
        </div>

        {/* Mobile Navigation */}
        {isMenuOpen && (
          <div className="md:hidden border-t border-border py-4">
            <div className="flex flex-col gap-4">
              {navItems.map((item) => {
                const Icon = item.icon;
                return (
                  <Link
                    key={item.href}
                    href={item.href}
                    className="flex items-center gap-3 text-foreground/80 hover:text-primary transition-colors font-medium py-2"
                    onClick={() => setIsMenuOpen(false)}
                  >
                    <Icon className="w-4 h-4" />
                    {item.label}
                  </Link>
                );
              })}
            </div>
          </div>
        )}
      </div>
    </nav>
  );
}
