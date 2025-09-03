"use client";

import { useCallback, useMemo, useState } from "react";
import { Button } from "./ui/button";
import { Input } from "./ui/input";
import { Textarea } from "./ui/textarea";
import { Label } from "./ui/label";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { Mail, Phone, MapPin, Send } from "lucide-react";

interface FormData {
  name: string;
  email: string;
  subject: string;
  message: string;
}

export default function ContactForm() {
  const [formData, setFormData] = useState<FormData>({
    name: "",
    email: "",
    subject: "",
    message: "",
  });
  const [errors, setErrors] = useState<Partial<Record<keyof FormData, string>>>(
    {},
  );
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submitStatus, setSubmitStatus] = useState<
    "idle" | "success" | "error"
  >("idle");
  const [hasSubmitted, setHasSubmitted] = useState(false);

  const emailRegex = useMemo(
    () =>
      // simple RFC5322-like email check
      /^[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}$/i,
    [],
  );

  const validate = useCallback(
    (values: FormData) => {
      const newErrors: Partial<Record<keyof FormData, string>> = {};

      if (!values.name || values.name.trim().length < 2) {
        newErrors.name = "Please enter your full name (at least 2 characters).";
      }

      if (!values.email || !emailRegex.test(values.email)) {
        newErrors.email = "Please enter a valid email address.";
      }

      if (!values.subject || values.subject.trim().length < 5) {
        newErrors.subject = "Please enter a subject (at least 5 characters).";
      }

      if (!values.message || values.message.trim().length < 20) {
        newErrors.message =
          "Please provide a detailed message (at least 20 characters).";
      }

      return newErrors;
    },
    [emailRegex],
  );

  const isFormValid = useMemo(
    () => Object.keys(validate(formData)).length === 0,
    [formData, validate],
  );

  const handleInputChange = (
    e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>,
  ) => {
    const { name, value } = e.target as { name: keyof FormData; value: string };
    setFormData((prev) => ({ ...prev, [name]: value }));

    // live-validate field and clear error as user types
    setErrors((prev) => {
      const next = { ...prev };
      // revalidate only this field
      const fieldErrors = validate({ ...formData, [name]: value });
      // if this field has no error after change, remove any previous
      if (!fieldErrors[name]) {
        delete next[name];
      } else {
        next[name] = fieldErrors[name];
      }
      return next;
    });
  };

  const handleBlur = (
    e: React.FocusEvent<HTMLInputElement | HTMLTextAreaElement>,
  ) => {
    const { name } = e.target as { name: keyof FormData };
    const fieldErrors = validate(formData);
    setErrors((prev) => ({ ...prev, [name]: fieldErrors[name] }));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setHasSubmitted(true);
    setSubmitStatus("idle");

    const foundErrors = validate(formData);
    setErrors(foundErrors);
    if (Object.keys(foundErrors).length > 0) {
      // invalid form, abort submit
      return;
    }

    setIsSubmitting(true);

    try {
      const response = await fetch("/api/contact", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(formData),
      });

      const result = await response.json();

      if (response.ok) {
        setSubmitStatus("success");
        setFormData({ name: "", email: "", subject: "", message: "" });
        setErrors({});
        setHasSubmitted(false);
      } else {
        throw new Error(result.error || "Failed to send message");
      }
    } catch (error) {
      console.error("Error submitting form:", error);
      setSubmitStatus("error");
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="bg-background min-h-screen py-20">
      <div className="container mx-auto px-4">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-12">
            <h1 className="text-4xl font-bold text-foreground mb-4 brand-wordmark">
              Get In Touch
            </h1>
            <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
              Ready to discuss your next Python development project? I'd love to
              hear from you. Let's build something amazing together with clean,
              efficient code and innovative solutions.
            </p>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-12">
            {/* Contact Information */}
            <div className="space-y-8">
              <Card>
                <CardHeader>
                  <CardTitle className="text-2xl">
                    Contact Information
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-6">
                  <div className="flex items-center gap-4">
                    <div className="bg-brand-aqua/10 p-3 rounded-full">
                      <Mail className="w-6 h-6 text-brand-aqua" />
                    </div>
                    <div>
                      <h3 className="font-semibold text-foreground">Email</h3>
                      <p className="text-muted-foreground">
                        contact@heliotheanalyst.co.uk
                      </p>
                    </div>
                  </div>

                  <div className="flex items-center gap-4">
                    <div className="bg-brand-plum/10 p-3 rounded-full">
                      <Phone className="w-6 h-6 text-brand-plum" />
                    </div>
                    <div>
                      <h3 className="font-semibold text-foreground">Phone</h3>
                      <p className="text-muted-foreground">+44 7818 351 949</p>
                    </div>
                  </div>

                  <div className="flex items-center gap-4">
                    <div className="bg-brand-aqua/10 p-3 rounded-full">
                      <MapPin className="w-6 h-6 text-brand-aqua" />
                    </div>
                    <div>
                      <h3 className="font-semibold text-foreground">
                        Location
                      </h3>
                      <p className="text-muted-foreground">Southampton, UK</p>
                    </div>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>What I Can Help With</CardTitle>
                </CardHeader>
                <CardContent>
                  <ul className="space-y-3 text-muted-foreground">
                    <li className="flex items-center gap-2">
                      <div className="w-2 h-2 bg-brand-aqua rounded-full"></div>
                      Python Development & API Integration
                    </li>
                    <li className="flex items-center gap-2">
                      <div className="w-2 h-2 bg-brand-aqua rounded-full"></div>
                      Shopify API & E-commerce Solutions
                    </li>
                    <li className="flex items-center gap-2">
                      <div className="w-2 h-2 bg-brand-aqua rounded-full"></div>
                      Web Scraping & Data Extraction
                    </li>
                    <li className="flex items-center gap-2">
                      <div className="w-2 h-2 bg-brand-aqua rounded-full"></div>
                      Process Automation & Selenium
                    </li>
                    <li className="flex items-center gap-2">
                      <div className="w-2 h-2 bg-brand-aqua rounded-full"></div>
                      Custom Software Development
                    </li>
                  </ul>
                </CardContent>
              </Card>
            </div>

            {/* Contact Form */}
            <Card>
              <CardHeader>
                <CardTitle className="text-2xl">Send Me a Message</CardTitle>
              </CardHeader>
              <CardContent>
                <form onSubmit={handleSubmit} className="space-y-6" noValidate>
                  {!isFormValid && hasSubmitted && (
                    <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 text-red-800 dark:text-red-200 px-4 py-3 rounded-md">
                      <div className="flex items-center gap-2">
                        <div className="w-2 h-2 bg-red-500 rounded-full"></div>
                        Please fix the errors below and try again.
                      </div>
                    </div>
                  )}

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <Label htmlFor="name">Name *</Label>
                      <Input
                        id="name"
                        name="name"
                        type="text"
                        value={formData.name}
                        onChange={handleInputChange}
                        onBlur={handleBlur}
                        aria-invalid={!!errors.name}
                        placeholder="Your full name"
                        className={
                          errors.name
                            ? "border-destructive focus-visible:ring-destructive"
                            : ""
                        }
                        required
                        minLength={2}
                      />
                      {errors.name && (
                        <p className="text-sm text-destructive">
                          {errors.name}
                        </p>
                      )}
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="email">Email *</Label>
                      <Input
                        id="email"
                        name="email"
                        type="email"
                        value={formData.email}
                        onChange={handleInputChange}
                        onBlur={handleBlur}
                        aria-invalid={!!errors.email}
                        placeholder="your.email@example.com"
                        className={
                          errors.email
                            ? "border-destructive focus-visible:ring-destructive"
                            : ""
                        }
                        required
                      />
                      {errors.email && (
                        <p className="text-sm text-destructive">
                          {errors.email}
                        </p>
                      )}
                    </div>
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="subject">Subject *</Label>
                    <Input
                      id="subject"
                      name="subject"
                      type="text"
                      value={formData.subject}
                      onChange={handleInputChange}
                      onBlur={handleBlur}
                      aria-invalid={!!errors.subject}
                      placeholder="What's this about?"
                      className={
                        errors.subject
                          ? "border-destructive focus-visible:ring-destructive"
                          : ""
                      }
                      required
                      minLength={5}
                    />
                    {errors.subject && (
                      <p className="text-sm text-destructive">
                        {errors.subject}
                      </p>
                    )}
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="message">Message *</Label>
                    <Textarea
                      id="message"
                      name="message"
                      value={formData.message}
                      onChange={handleInputChange}
                      onBlur={handleBlur}
                      aria-invalid={!!errors.message}
                      placeholder="Tell me about your project or question..."
                      rows={6}
                      className={
                        errors.message
                          ? "border-destructive focus-visible:ring-destructive"
                          : ""
                      }
                      required
                      minLength={20}
                    />
                    {errors.message && (
                      <p className="text-sm text-destructive">
                        {errors.message}
                      </p>
                    )}
                  </div>

                  {submitStatus === "success" && (
                    <div className="bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 text-green-800 dark:text-green-200 px-4 py-3 rounded-md">
                      <div className="flex items-center gap-2">
                        <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                        Thank you for your message! I'll get back to you within
                        24 hours.
                      </div>
                    </div>
                  )}

                  {submitStatus === "error" && (
                    <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 text-red-800 dark:text-red-200 px-4 py-3 rounded-md">
                      <div className="flex items-center gap-2">
                        <div className="w-2 h-2 bg-red-500 rounded-full"></div>
                        Sorry, there was an error sending your message. Please
                        try again or contact me directly.
                      </div>
                    </div>
                  )}

                  <Button
                    type="submit"
                    disabled={isSubmitting || !isFormValid}
                    className="w-full flex items-center justify-center gap-2"
                  >
                    {isSubmitting ? (
                      "Sending..."
                    ) : (
                      <>
                        <Send className="w-4 h-4" />
                        Send Message
                      </>
                    )}
                  </Button>
                </form>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </div>
  );
}
