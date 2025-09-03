"use client";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Download,
  Mail,
  Phone,
  MapPin,
  Calendar,
  Award,
  Briefcase,
  GraduationCap,
  Code,
  Database,
  BarChart3,
} from "lucide-react";

interface Experience {
  title: string;
  company: string;
  period: string;
  description: string[];
  technologies: string[];
}

interface Education {
  degree: string;
  institution: string;
  period: string;
  details?: string;
}

interface Certification {
  name: string;
  issuer: string;
  date: string;
}

const experiences: Experience[] = [
  {
    title: "Software Engineer",
    company: "Paul Murray Plc",
    period: "2023 - 2024",
    description: [
      "Automated the integration of API endpoints with Shopify, reducing manual input time by over 15 hours per month for product updates",
      "Improved synchronisation speed by 60% through API integrations with systems like BC365 and Shopify, enhancing operational efficiency",
      "Designed and implemented a web scraping tool with Selenium, extracting data from 500+ competitor websites to support strategic product adjustments",
      "Increased order processing accuracy and boosted monthly orders by 300+ through Shopify API integration",
      "Collaborated with cross-functional teams during sprints to identify and address three major workflow inefficiencies based on user feedback",
      "Created internal API documentation, enabling smooth onboarding of four new developers into ongoing projects",
      "Conducted workshops for team members on using API tools and automation techniques, improving knowledge sharing across departments",
      "Managed a project to integrate advanced error-handling mechanisms into API workflows, reducing downtime by 20%",
    ],
    technologies: [
      "Python",
      "Selenium",
      "Shopify API",
      "BC365",
      "SQL",
      "API Integration",
    ],
  },
  {
    title: "Restaurant Manager",
    company: "Burping Ron's",
    period: "2023 - 2023",
    description: [
      "Managed daily operations, including staff coordination and process improvement to enhance customer satisfaction",
      "Implemented a new digital inventory tracking system, reducing food waste by 15%",
    ],
    technologies: [
      "Digital Systems",
      "Process Optimization",
      "Team Management",
    ],
  },
  {
    title: "Restaurant Manager",
    company: "Rosewood Investment Limited",
    period: "2021 - 2023",
    description: [
      "Directed operational processes, implementing initiatives that enhanced customer retention and improved service quality",
      "Mentored and coached a team of 10+ employees, driving a 20% improvement in service delivery times",
      "Streamlined recruitment and selection processes, achieving a 30% increase in efficiency",
    ],
    technologies: [
      "Team Leadership",
      "Process Improvement",
      "Operations Management",
    ],
  },
];

const education: Education[] = [
  {
    degree: "Data Analyst Certification",
    institution: "Masterschool, London, UK",
    period: "2023 - 2024",
    details:
      "Coursework: Python programming, advanced SQL, data visualisation, machine learning concepts",
  },
];

const certifications: Certification[] = [
  {
    name: "Data Analyst Certification",
    issuer: "Masterschool",
    date: "2024",
  },
];

const skills = {
  "Programming Languages": ["Python", "SQL", "JavaScript", "CSS"],
  "Web Scraping & Automation": [
    "Selenium",
    "API Integration",
    "Data Extraction",
    "Process Automation",
  ],
  "Data Analysis": [
    "Pandas",
    "NumPy",
    "Data Visualization",
    "Statistical Analysis",
  ],
  "Systems & Platforms": ["Shopify API", "BC365", "Linux", "Docker"],
  Cybersecurity: ["Ethical Hacking", "Penetration Testing", "Network Security"],
  Networking: ["TCP/IP", "DHCP", "DNS", "Cisco", "Web Server"],
};

export default function CVPage() {
  const handleDownloadCV = () => {
    // In a real application, this would download an actual PDF file
    // For now, we'll create a simple alert
    alert(
      "CV download would start here. In a real application, this would download a PDF file.",
    );
  };

  return (
    <div className="bg-background min-h-screen transition-colors duration-300">
      {/* Header */}
      <section className="brand-gradient text-white py-16">
        <div className="container mx-auto px-4">
          <div className="max-w-4xl mx-auto text-center">
            <div className="flex justify-center mb-8">
              <div className="logo-mark">
                <svg
                  width="100"
                  height="100"
                  viewBox="0 0 120 120"
                  className="drop-shadow-lg"
                >
                  {/* Outer ring with gap */}
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
                  {/* Inner H */}
                  <g fill="white">
                    <rect x="42" y="35" width="6" height="50" />
                    <rect x="72" y="35" width="6" height="50" />
                    <rect x="42" y="57" width="36" height="6" />
                  </g>
                </svg>
              </div>
            </div>
            <div className="mb-6">
              <h1 className="brand-wordmark text-5xl font-semibold mb-2 text-white drop-shadow-sm transition-colors duration-300">
                Helio Bruno Garcia Massadico
              </h1>
              <p className="brand-tagline text-sm tracking-[0.18em] text-white/90 mb-4 transition-colors duration-300">
                Mid-Level Python Developer & Data Analyst
              </p>
            </div>
            <div className="flex flex-wrap justify-center gap-6 mb-8 text-lg">
              <div className="flex items-center gap-2">
                <Mail className="w-5 h-5" />
                <span>contact@heliotheanalyst.co.uk</span>
              </div>
              <div className="flex items-center gap-2">
                <Phone className="w-5 h-5" />
                <span>+44 7818 351 949</span>
              </div>
              <div className="flex items-center gap-2">
                <MapPin className="w-5 h-5" />
                <span>Southampton, UK</span>
              </div>
            </div>
            <Button
              onClick={handleDownloadCV}
              size="lg"
              className="bg-white text-brand-plum hover:bg-white/90 flex items-center gap-2 mx-auto font-medium"
            >
              <Download className="w-5 h-5" />
              Download CV (PDF)
            </Button>
          </div>
        </div>
      </section>

      <div className="container mx-auto px-4 py-16">
        <div className="max-w-6xl mx-auto space-y-12">
          {/* Professional Summary */}
          <Card className="transition-all duration-300">
            <CardHeader>
              <CardTitle className="text-2xl flex items-center gap-2 brand-wordmark transition-colors duration-300">
                <BarChart3 className="w-6 h-6 text-brand-aqua transition-colors duration-300" />
                Professional Summary
              </CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-lg text-muted-foreground leading-relaxed transition-colors duration-300">
                Mid-level Python Developer with 5+ years of experience in API
                integration, automation, and data processing. Specialized in
                Shopify API integrations that reduced manual work by 15+ hours
                per month and improved synchronization speed by 60%. Expert in
                web scraping with Selenium, extracting data from 500+ competitor
                websites for strategic analysis. Developed automated systems
                that increased order processing accuracy and boosted monthly
                orders by 300+. Proficient in Python, SQL, Selenium, and API
                development, with a strong focus on process optimization and
                delivering measurable business results. All portfolio projects
                demonstrate hands-on development and problem-solving skills.
              </p>
            </CardContent>
          </Card>

          {/* Experience */}
          <Card className="transition-all duration-300">
            <CardHeader>
              <CardTitle className="text-2xl flex items-center gap-2 brand-wordmark transition-colors duration-300">
                <Briefcase className="w-6 h-6 text-brand-aqua transition-colors duration-300" />
                Professional Experience
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-8">
              {experiences.map((exp, index) => (
                <div
                  key={index}
                  className="border-l-4 border-primary/30 pl-6 transition-all duration-300"
                >
                  <div className="flex flex-col md:flex-row md:items-center md:justify-between mb-2">
                    <h3 className="text-xl font-semibold text-foreground transition-colors duration-300">
                      {exp.title}
                    </h3>
                    <div className="flex items-center gap-2 text-muted-foreground transition-colors duration-300">
                      <Calendar className="w-4 h-4" />
                      <span>{exp.period}</span>
                    </div>
                  </div>
                  <p className="text-lg text-primary font-medium mb-3 transition-colors duration-300">
                    {exp.company}
                  </p>
                  <ul className="space-y-2 mb-4">
                    {exp.description.map((item, idx) => (
                      <li
                        key={idx}
                        className="text-muted-foreground flex items-start gap-2 transition-colors duration-300"
                      >
                        <div className="w-2 h-2 bg-primary rounded-full mt-2 flex-shrink-0 transition-colors duration-300"></div>
                        <span>{item}</span>
                      </li>
                    ))}
                  </ul>
                  <div className="flex flex-wrap gap-2">
                    {exp.technologies.map((tech) => (
                      <span
                        key={tech}
                        className="px-3 py-1 bg-primary/10 text-primary text-sm rounded-full transition-all duration-300"
                      >
                        {tech}
                      </span>
                    ))}
                  </div>
                </div>
              ))}
            </CardContent>
          </Card>

          {/* Skills */}
          <Card className="transition-all duration-300">
            <CardHeader>
              <CardTitle className="text-2xl flex items-center gap-2 brand-wordmark transition-colors duration-300">
                <Code className="w-6 h-6 text-brand-aqua transition-colors duration-300" />
                Technical Skills
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {Object.entries(skills).map(([category, skillList]) => (
                  <div key={category}>
                    <h3 className="font-semibold text-foreground mb-3 flex items-center gap-2 transition-colors duration-300">
                      <Database className="w-4 h-4 text-primary transition-colors duration-300" />
                      {category}
                    </h3>
                    <div className="flex flex-wrap gap-2">
                      {skillList.map((skill) => (
                        <span
                          key={skill}
                          className="px-2 py-1 bg-muted text-muted-foreground text-sm rounded transition-all duration-300"
                        >
                          {skill}
                        </span>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          {/* Education */}
          <Card className="transition-all duration-300">
            <CardHeader>
              <CardTitle className="text-2xl flex items-center gap-2 brand-wordmark transition-colors duration-300">
                <GraduationCap className="w-6 h-6 text-brand-aqua transition-colors duration-300" />
                Education
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              {education.map((edu, index) => (
                <div
                  key={index}
                  className="border-l-4 border-accent/30 pl-6 transition-all duration-300"
                >
                  <div className="flex flex-col md:flex-row md:items-center md:justify-between mb-2">
                    <h3 className="text-xl font-semibold text-foreground transition-colors duration-300">
                      {edu.degree}
                    </h3>
                    <div className="flex items-center gap-2 text-muted-foreground transition-colors duration-300">
                      <Calendar className="w-4 h-4" />
                      <span>{edu.period}</span>
                    </div>
                  </div>
                  <p className="text-lg text-accent font-medium mb-2 transition-colors duration-300">
                    {edu.institution}
                  </p>
                  {edu.details && (
                    <p className="text-muted-foreground transition-colors duration-300">
                      {edu.details}
                    </p>
                  )}
                </div>
              ))}
            </CardContent>
          </Card>

          {/* Certifications */}
          <Card className="transition-all duration-300">
            <CardHeader>
              <CardTitle className="text-2xl flex items-center gap-2 brand-wordmark transition-colors duration-300">
                <Award className="w-6 h-6 text-brand-aqua transition-colors duration-300" />
                Certifications
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {certifications.map((cert, index) => (
                  <div
                    key={index}
                    className="bg-muted p-4 rounded-lg transition-all duration-300"
                  >
                    <h3 className="font-semibold text-foreground mb-1 transition-colors duration-300">
                      {cert.name}
                    </h3>
                    <p className="text-muted-foreground mb-1 transition-colors duration-300">
                      {cert.issuer}
                    </p>
                    <p className="text-sm text-muted-foreground transition-colors duration-300">
                      {cert.date}
                    </p>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          {/* Call to Action */}
          <Card className="bg-gradient-to-r from-brand-aqua/10 to-brand-plum/10 transition-all duration-300">
            <CardContent className="text-center py-8">
              <h2 className="text-2xl font-bold text-foreground mb-4 brand-wordmark transition-colors duration-300">
                Ready to Collaborate?
              </h2>
              <p className="text-lg text-muted-foreground mb-6 max-w-2xl mx-auto transition-colors duration-300">
                I'm always interested in discussing new opportunities and
                challenging data problems. Let's connect and explore how we can
                work together.
              </p>
              <div className="flex flex-col sm:flex-row gap-4 justify-center">
                <Button size="lg" asChild>
                  <a href="/contact">Get In Touch</a>
                </Button>
                <Button variant="outline" size="lg" onClick={handleDownloadCV}>
                  <Download className="w-4 h-4 mr-2" />
                  Download Full CV
                </Button>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
