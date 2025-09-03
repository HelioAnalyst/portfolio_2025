import { notFound } from "next/navigation";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "../../../components/ui/card";
import { Button } from "../../../components/ui/button";
import { Badge } from "../../../components/ui/badge";
import {
  ExternalLink,
  Github,
  ArrowLeft,
  Calendar,
  Users,
  Target,
  Zap,
} from "lucide-react";
import Link from "next/link";

interface Project {
  id: string;
  title: string;
  description: string;
  technologies: string[];
  image: string;
  githubUrl?: string;
  liveUrl?: string;
  fullDescription: string;
  challenges: string[];
  solutions: string[];
  results: string[];
  duration: string;
  teamSize: string;
  role: string;
}

const projects: Project[] = [
  {
    id: "1",
    title: "Shopify API Integration System",
    description:
      "Automated API integration system for Shopify that reduced manual product update time by 15+ hours per month and improved synchronization speed by 60%.",
    fullDescription:
      "Developed a comprehensive Shopify API integration system that automates product management, inventory synchronization, and order processing. The system features robust error handling, retry logic with exponential backoff, and comprehensive logging. It processes bulk product updates, manages inventory levels across multiple locations, and synchronizes data with external systems including BC365. The solution significantly reduced manual work while improving data accuracy and processing speed.",
    technologies: [
      "Python",
      "Shopify API",
      "BC365",
      "SQL",
      "FastAPI",
      "PostgreSQL",
      "Redis",
      "Docker",
    ],
    image:
      "https://images.unsplash.com/photo-1556742049-0cfed4f6a45d?w=800&q=80",
    githubUrl: "https://github.com/helio/shopify-integration",
    liveUrl: "https://shopify-demo.helio.com",
    challenges: [
      "Managing API rate limits and implementing proper throttling",
      "Handling large product catalogs with thousands of SKUs",
      "Ensuring data consistency between Shopify and external systems",
      "Implementing robust error recovery for failed API calls",
    ],
    solutions: [
      "Implemented exponential backoff retry logic for API calls",
      "Created batch processing system for bulk operations",
      "Built comprehensive logging and monitoring system",
      "Developed automated data validation and reconciliation processes",
    ],
    results: [
      "Reduced manual product update time by 15+ hours per month",
      "Improved synchronization speed by 60%",
      "Achieved 99.5% data accuracy across systems",
      "Processed 10,000+ product updates daily",
    ],
    duration: "6 months",
    teamSize: "3 developers",
    role: "Lead Python Developer & API Integration Specialist",
  },
  {
    id: "2",
    title: "Web Scraping & Data Extraction Tool",
    description:
      "Selenium-based web scraping tool that extracts data from 500+ competitor websites to support strategic product adjustments and market analysis.",
    fullDescription:
      "Built a comprehensive web scraping system using Selenium that extracts competitor product data, pricing information, and market intelligence from over 500 websites. The system features dynamic content handling, pagination support, anti-detection measures, and robust error handling. It includes data validation, quality checks, and automated reporting capabilities that provide actionable insights for strategic business decisions.",
    technologies: [
      "Python",
      "Selenium",
      "BeautifulSoup",
      "Pandas",
      "SQLite",
      "Chrome WebDriver",
      "Requests",
      "JSON",
    ],
    image:
      "https://images.unsplash.com/photo-1518186285589-2f7649de83e0?w=800&q=80",
    githubUrl: "https://github.com/helio/web-scraper",
    liveUrl: "https://scraper-demo.helio.com",
    challenges: [
      "Handling dynamic content and JavaScript-heavy websites",
      "Implementing anti-detection measures to avoid blocking",
      "Managing different website structures and layouts",
      "Ensuring data quality and consistency across sources",
    ],
    solutions: [
      "Used Selenium with headless Chrome for dynamic content",
      "Implemented rotating user agents and request delays",
      "Created configurable selectors for different site structures",
      "Built comprehensive data validation and cleaning pipeline",
    ],
    results: [
      "Successfully scraped data from 500+ competitor websites",
      "Provided market intelligence for strategic product adjustments",
      "Achieved 95% data extraction accuracy",
      "Generated automated competitive analysis reports",
    ],
    duration: "4 months",
    teamSize: "2 developers",
    role: "Senior Python Developer & Web Scraping Specialist",
  },
  {
    id: "3",
    title: "Order Processing Automation",
    description:
      "Automated order processing system that increased accuracy and boosted monthly orders by 300+ through intelligent API integrations and error handling.",
    fullDescription:
      "Developed an intelligent order processing automation system that streamlines the entire order lifecycle from placement to fulfillment. The system integrates with multiple APIs, implements smart routing logic, and features comprehensive error handling with automated recovery mechanisms. It includes real-time monitoring, performance analytics, and automated reporting capabilities that significantly improved processing efficiency and order accuracy.",
    technologies: [
      "Python",
      "FastAPI",
      "PostgreSQL",
      "Redis",
      "Docker",
      "Celery",
      "SQLAlchemy",
      "Pydantic",
    ],
    image:
      "https://images.unsplash.com/photo-1460925895917-afdab827c52f?w=800&q=80",
    githubUrl: "https://github.com/helio/order-automation",
    liveUrl: "https://orders-demo.helio.com",
    challenges: [
      "Managing complex order workflows with multiple dependencies",
      "Ensuring data integrity across distributed systems",
      "Handling high-volume order processing during peak times",
      "Implementing real-time error detection and recovery",
    ],
    solutions: [
      "Built asynchronous processing pipeline with Celery",
      "Implemented database transactions for data consistency",
      "Created auto-scaling infrastructure with Docker",
      "Developed comprehensive monitoring and alerting system",
    ],
    results: [
      "Increased monthly order processing by 300+",
      "Improved order accuracy to 99.8%",
      "Reduced processing time by 70%",
      "Achieved 99.9% system uptime",
    ],
    duration: "5 months",
    teamSize: "4 developers",
    role: "Lead Backend Developer & Automation Specialist",
  },
  {
    id: "4",
    title: "Data Analysis & Visualization Suite",
    description:
      "Comprehensive Python toolkit for data cleaning, statistical analysis, and visualization using pandas, numpy, and matplotlib for business insights.",
    fullDescription:
      "Created a comprehensive data analysis toolkit that automates the entire analytics workflow from data ingestion to insight generation. The suite includes advanced data cleaning algorithms, statistical analysis modules, and interactive visualization components. It features automated report generation, anomaly detection, and predictive analytics capabilities that enable data-driven decision making across the organization.",
    technologies: [
      "Python",
      "Pandas",
      "NumPy",
      "Matplotlib",
      "Seaborn",
      "Jupyter",
      "Plotly",
      "Streamlit",
    ],
    image:
      "https://images.unsplash.com/photo-1526379095098-d400fd0bf935?w=800&q=80",
    githubUrl: "https://github.com/helio/data-analysis",
    liveUrl: "https://analysis-demo.helio.com",
    challenges: [
      "Handling diverse data formats and quality issues",
      "Creating reusable analysis templates for different use cases",
      "Building interactive visualizations for non-technical users",
      "Implementing statistical validation and significance testing",
    ],
    solutions: [
      "Developed modular data cleaning and preprocessing pipeline",
      "Created template library with parameterized Jupyter notebooks",
      "Built interactive dashboards using Plotly and Streamlit",
      "Implemented comprehensive statistical testing framework",
    ],
    results: [
      "Reduced data analysis time by 60%",
      "Standardized analytics processes across teams",
      "Generated automated insights for business stakeholders",
      "Enabled self-service analytics for 25+ users",
    ],
    duration: "6 months",
    teamSize: "3 analysts",
    role: "Senior Data Analyst & Python Developer",
  },
  {
    id: "5",
    title: "API Documentation & Testing Framework",
    description:
      "Internal API documentation system and testing framework that enabled smooth onboarding of new developers and reduced integration errors by 20%.",
    fullDescription:
      "Built a comprehensive API documentation and testing framework that streamlines developer onboarding and ensures API reliability. The system features automated documentation generation, interactive API testing, comprehensive test suites, and performance monitoring. It includes code examples, integration guides, and automated validation that significantly improved developer experience and reduced integration errors.",
    technologies: [
      "Python",
      "FastAPI",
      "Swagger",
      "Pytest",
      "Docker",
      "OpenAPI",
      "Postman",
      "GitHub Actions",
    ],
    image:
      "https://images.unsplash.com/photo-1555949963-aa79dcee981c?w=800&q=80",
    githubUrl: "https://github.com/helio/api-docs",
    liveUrl: "https://api-docs-demo.helio.com",
    challenges: [
      "Creating comprehensive documentation for complex APIs",
      "Implementing automated testing for multiple endpoints",
      "Ensuring documentation stays synchronized with code changes",
      "Building user-friendly testing interfaces for developers",
    ],
    solutions: [
      "Implemented automated documentation generation with FastAPI",
      "Created comprehensive test suites with Pytest",
      "Built CI/CD pipeline for automated testing and deployment",
      "Developed interactive API explorer with Swagger UI",
    ],
    results: [
      "Reduced developer onboarding time by 50%",
      "Decreased API integration errors by 20%",
      "Improved API test coverage to 95%",
      "Enabled faster development cycles for new features",
    ],
    duration: "3 months",
    teamSize: "2 developers",
    role: "API Developer & Documentation Specialist",
  },
  {
    id: "6",
    title: "Digital Inventory Management System",
    description:
      "Digital inventory tracking system that reduced food waste by 15% through automated monitoring and predictive analytics for restaurant operations.",
    fullDescription:
      "Developed a comprehensive digital inventory management system specifically designed for restaurant operations. The system features real-time inventory tracking, automated reorder alerts, expiration date monitoring, and predictive analytics for demand forecasting. It includes waste tracking, cost analysis, and reporting capabilities that help optimize inventory levels and reduce food waste while maintaining operational efficiency.",
    technologies: [
      "Python",
      "SQLite",
      "Pandas",
      "Tkinter",
      "Matplotlib",
      "NumPy",
      "Datetime",
      "CSV",
    ],
    image:
      "https://images.unsplash.com/photo-1586528116311-ad8dd3c8310d?w=800&q=80",
    githubUrl: "https://github.com/helio/inventory-system",
    liveUrl: "https://inventory-demo.helio.com",
    challenges: [
      "Tracking perishable items with varying shelf lives",
      "Implementing predictive analytics for demand forecasting",
      "Creating user-friendly interface for restaurant staff",
      "Ensuring real-time accuracy of inventory data",
    ],
    solutions: [
      "Built automated expiration tracking with alert system",
      "Implemented machine learning models for demand prediction",
      "Created intuitive GUI with Tkinter for easy operation",
      "Developed real-time synchronization with barcode scanning",
    ],
    results: [
      "Reduced food waste by 15% through better tracking",
      "Improved inventory accuracy to 98%",
      "Decreased ordering costs by 12%",
      "Enabled data-driven inventory decisions",
    ],
    duration: "4 months",
    teamSize: "2 developers",
    role: "Python Developer & System Analyst",
  },
];

export default function ProjectPage({ params }: { params: { id: string } }) {
  const project = projects.find((p) => p.id === params.id);

  if (!project) {
    notFound();
  }

  return (
    <div className="bg-background min-h-screen">
      {/* Header */}
      <div className="brand-gradient text-white py-12">
        <div className="container mx-auto px-4">
          <Link
            href="/"
            className="inline-flex items-center gap-2 text-white/80 hover:text-white mb-6"
          >
            <ArrowLeft className="w-4 h-4" />
            Back to Portfolio
          </Link>
          <h1 className="text-4xl font-bold mb-4">{project.title}</h1>
          <p className="text-xl text-white/90 max-w-3xl">
            {project.description}
          </p>
        </div>
      </div>

      <div className="container mx-auto px-4 py-12">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Main Content */}
          <div className="lg:col-span-2 space-y-8">
            {/* Project Image */}
            <div className="aspect-video overflow-hidden rounded-lg">
              <img
                src={project.image}
                alt={project.title}
                className="w-full h-full object-cover"
              />
            </div>

            {/* Project Overview */}
            <Card>
              <CardHeader>
                <CardTitle>Project Overview</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-muted-foreground leading-relaxed">
                  {project.fullDescription}
                </p>
              </CardContent>
            </Card>

            {/* Challenges */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Target className="w-5 h-5" />
                  Challenges
                </CardTitle>
              </CardHeader>
              <CardContent>
                <ul className="space-y-3">
                  {project.challenges.map((challenge, index) => (
                    <li key={index} className="flex items-start gap-3">
                      <div className="w-2 h-2 bg-red-500 rounded-full mt-2 flex-shrink-0" />
                      <span className="text-muted-foreground">{challenge}</span>
                    </li>
                  ))}
                </ul>
              </CardContent>
            </Card>

            {/* Solutions */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Zap className="w-5 h-5" />
                  Solutions
                </CardTitle>
              </CardHeader>
              <CardContent>
                <ul className="space-y-3">
                  {project.solutions.map((solution, index) => (
                    <li key={index} className="flex items-start gap-3">
                      <div className="w-2 h-2 bg-green-500 rounded-full mt-2 flex-shrink-0" />
                      <span className="text-muted-foreground">{solution}</span>
                    </li>
                  ))}
                </ul>
              </CardContent>
            </Card>

            {/* Results */}
            <Card>
              <CardHeader>
                <CardTitle>Results & Impact</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {project.results.map((result, index) => (
                    <div
                      key={index}
                      className="bg-green-50 p-4 rounded-lg border border-green-200"
                    >
                      <span className="text-green-800 font-medium">
                        {result}
                      </span>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Sidebar */}
          <div className="space-y-6">
            {/* Project Details */}
            <Card>
              <CardHeader>
                <CardTitle>Project Details</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex items-center gap-3">
                  <Calendar className="w-5 h-5 text-muted-foreground" />
                  <div>
                    <p className="font-medium">Duration</p>
                    <p className="text-muted-foreground">{project.duration}</p>
                  </div>
                </div>
                <div className="flex items-center gap-3">
                  <Users className="w-5 h-5 text-muted-foreground" />
                  <div>
                    <p className="font-medium">Team Size</p>
                    <p className="text-muted-foreground">{project.teamSize}</p>
                  </div>
                </div>
                <div>
                  <p className="font-medium mb-2">My Role</p>
                  <p className="text-muted-foreground">{project.role}</p>
                </div>
              </CardContent>
            </Card>

            {/* Technologies */}
            <Card>
              <CardHeader>
                <CardTitle>Technologies Used</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="flex flex-wrap gap-2">
                  {project.technologies.map((tech) => (
                    <Badge key={tech} variant="secondary">
                      {tech}
                    </Badge>
                  ))}
                </div>
              </CardContent>
            </Card>

            {/* Links */}
            <Card>
              <CardHeader>
                <CardTitle>Project Links</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                {project.githubUrl && (
                  <Button
                    variant="outline"
                    className="w-full justify-start"
                    asChild
                  >
                    <a
                      href={project.githubUrl}
                      target="_blank"
                      rel="noopener noreferrer"
                    >
                      <Github className="w-4 h-4 mr-2" />
                      View Source Code
                    </a>
                  </Button>
                )}
                {project.liveUrl && (
                  <Button className="w-full justify-start" asChild>
                    <a
                      href={project.liveUrl}
                      target="_blank"
                      rel="noopener noreferrer"
                    >
                      <ExternalLink className="w-4 h-4 mr-2" />
                      Live Demo
                    </a>
                  </Button>
                )}
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </div>
  );
}
