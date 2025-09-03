import { Card, CardContent } from "./ui/card";
import { Star, Quote } from "lucide-react";

interface Testimonial {
  id: string;
  name: string;
  role: string;
  company: string;
  content: string;
  rating: number;
  avatar: string;
}

const testimonials: Testimonial[] = [
  {
    id: "1",
    name: "Sarah Chen",
    role: "Operations Manager",
    company: "TechFlow Solutions",
    content:
      "Helio's Shopify API integration saved us 15+ hours per month on manual product updates. The system is robust, well-documented, and has significantly improved our workflow efficiency. His attention to detail and proactive communication made the project seamless.",
    rating: 5,
    avatar: "https://api.dicebear.com/7.x/avataaars/svg?seed=Sarah",
  },
  {
    id: "2",
    name: "Marcus Rodriguez",
    role: "Data Director",
    company: "Retail Analytics Co",
    content:
      "The web scraping solution Helio built for us extracts data from 500+ competitor websites with incredible accuracy. His expertise in Selenium and data processing has given us a competitive edge in market analysis. Highly recommend his services.",
    rating: 5,
    avatar: "https://api.dicebear.com/7.x/avataaars/svg?seed=Marcus",
  },
  {
    id: "3",
    name: "Emily Watson",
    role: "Restaurant Owner",
    company: "Green Leaf Bistro",
    content:
      "The inventory management system Helio developed reduced our food waste by 15% and streamlined our entire ordering process. His Python automation skills transformed how we manage our restaurant operations. Exceptional work and great communication throughout.",
    rating: 5,
    avatar: "https://api.dicebear.com/7.x/avataaars/svg?seed=Emily",
  },
  {
    id: "4",
    name: "David Kim",
    role: "CTO",
    company: "StartupHub",
    content:
      "Helio's order processing automation increased our monthly orders by 300+ while maintaining 99.9% accuracy. His FastAPI implementation is clean, scalable, and well-tested. He's become our go-to developer for Python automation projects.",
    rating: 5,
    avatar: "https://api.dicebear.com/7.x/avataaars/svg?seed=David",
  },
  {
    id: "5",
    name: "Lisa Thompson",
    role: "Business Analyst",
    company: "DataDriven Inc",
    content:
      "The data analysis toolkit Helio created using pandas and matplotlib has revolutionized our reporting process. His statistical analysis capabilities and visualization skills helped us uncover insights we never knew existed in our data.",
    rating: 5,
    avatar: "https://api.dicebear.com/7.x/avataaars/svg?seed=Lisa",
  },
  {
    id: "6",
    name: "James Park",
    role: "Product Manager",
    company: "InnovateTech",
    content:
      "Working with Helio on our API documentation and testing framework was fantastic. His attention to detail reduced our integration errors by 20% and made onboarding new developers much smoother. Professional, reliable, and skilled.",
    rating: 5,
    avatar: "https://api.dicebear.com/7.x/avataaars/svg?seed=James",
  },
];

export default function Testimonials() {
  return (
    <section className="py-20 bg-muted/30">
      <div className="container mx-auto px-4">
        <div className="text-center mb-12">
          <h2 className="text-4xl font-bold text-foreground mb-4 brand-wordmark">
            Client Testimonials
          </h2>
          <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
            Don't just take my word for it. Here's what clients say about
            working with me on their Python development and data analysis
            projects.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
          {testimonials.map((testimonial) => (
            <Card
              key={testimonial.id}
              className="h-full hover:shadow-lg transition-shadow"
            >
              <CardContent className="p-6 flex flex-col h-full">
                {/* Quote Icon */}
                <div className="mb-4">
                  <Quote className="w-8 h-8 text-brand-aqua opacity-60" />
                </div>

                {/* Rating */}
                <div className="flex gap-1 mb-4">
                  {[...Array(testimonial.rating)].map((_, i) => (
                    <Star
                      key={i}
                      className="w-4 h-4 fill-yellow-400 text-yellow-400"
                    />
                  ))}
                </div>

                {/* Content */}
                <p className="text-foreground/80 mb-6 flex-grow leading-relaxed">
                  &quot;{testimonial.content}&quot;
                </p>

                {/* Author */}
                <div className="flex items-center gap-4">
                  <img
                    src={testimonial.avatar}
                    alt={testimonial.name}
                    className="w-12 h-12 rounded-full bg-gray-200"
                  />
                  <div>
                    <h4 className="font-semibold text-foreground">
                      {testimonial.name}
                    </h4>
                    <p className="text-sm text-muted-foreground">
                      {testimonial.role}
                    </p>
                    <p className="text-sm text-brand-plum font-medium">
                      {testimonial.company}
                    </p>
                  </div>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>

        {/* Call to Action */}
        <div className="mt-16">
          <div className="bg-gradient-to-br from-brand-plum/10 via-brand-aqua/5 to-brand-plum/10 rounded-2xl p-8 md:p-12 border border-brand-plum/20 shadow-lg">
            <div className="text-center max-w-3xl mx-auto">
              <h3 className="text-3xl md:text-4xl font-bold text-foreground mb-4 brand-wordmark">
                Ready to Transform Your Business?
              </h3>
              <p className="text-lg md:text-xl text-muted-foreground mb-8 leading-relaxed">
                Join these satisfied clients and experience the power of custom
                Python solutions. Let's discuss how I can help streamline your
                operations and boost your productivity.
              </p>
              <div className="flex flex-col sm:flex-row gap-4 justify-center items-center">
                <a
                  href="/contact"
                  className="group inline-flex items-center justify-center px-8 py-4 bg-gradient-to-r from-brand-plum to-brand-plum/90 text-white font-semibold rounded-xl hover:from-brand-plum/90 hover:to-brand-plum shadow-lg hover:shadow-xl transform hover:-translate-y-0.5 transition-all duration-200"
                >
                  <span>Start Your Project</span>
                  <svg
                    className="ml-2 w-5 h-5 group-hover:translate-x-1 transition-transform"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M17 8l4 4m0 0l-4 4m4-4H3"
                    />
                  </svg>
                </a>
                <a
                  href="/cv"
                  className="group inline-flex items-center justify-center px-8 py-4 bg-background border-2 border-brand-aqua text-brand-aqua font-semibold rounded-xl hover:bg-brand-aqua hover:text-white shadow-md hover:shadow-lg transform hover:-translate-y-0.5 transition-all duration-200"
                >
                  <svg
                    className="mr-2 w-5 h-5"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
                    />
                  </svg>
                  <span>View My Experience</span>
                </a>
              </div>
              <div className="mt-8 flex items-center justify-center gap-6 text-sm text-muted-foreground">
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                  <span>Available for new projects</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 bg-brand-aqua rounded-full"></div>
                  <span>Quick response guaranteed</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
