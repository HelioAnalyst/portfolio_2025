import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Calendar, Clock, ArrowLeft, Share2, BookOpen } from "lucide-react";
import Link from "next/link";
import { notFound } from "next/navigation";
import { getPostById } from "@/lib/blog";
import SocialShareButtons from "@/components/SocialShareButtons";

function formatDate(dateString: string) {
  return new Date(dateString).toLocaleDateString("en-US", {
    year: "numeric",
    month: "long",
    day: "numeric",
  });
}

function renderMarkdownContent(content: string) {
  // Simple markdown-like rendering for demo purposes
  // In a real app, you'd use a proper markdown parser like react-markdown
  const lines = content.trim().split("\n");
  const elements: JSX.Element[] = [];

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];

    if (line.startsWith("# ")) {
      elements.push(
        <h1 key={i} className="text-3xl font-bold text-foreground mt-8 mb-4">
          {line.slice(2)}
        </h1>,
      );
    } else if (line.startsWith("## ")) {
      elements.push(
        <h2
          key={i}
          className="text-2xl font-semibold text-foreground mt-6 mb-3"
        >
          {line.slice(3)}
        </h2>,
      );
    } else if (line.startsWith("### ")) {
      elements.push(
        <h3 key={i} className="text-xl font-semibold text-foreground mt-4 mb-2">
          {line.slice(4)}
        </h3>,
      );
    } else if (line.startsWith("```")) {
      // Find the end of code block
      let codeContent = "";
      let j = i + 1;
      while (j < lines.length && !lines[j].startsWith("```")) {
        codeContent += lines[j] + "\n";
        j++;
      }
      elements.push(
        <pre key={i} className="bg-muted p-4 rounded-lg overflow-x-auto my-4">
          <code className="text-sm">{codeContent}</code>
        </pre>,
      );
      i = j; // Skip to end of code block
    } else if (line.trim() === "") {
      elements.push(<br key={i} />);
    } else {
      elements.push(
        <p key={i} className="text-foreground/90 leading-relaxed mb-4">
          {line}
        </p>,
      );
    }
  }

  return elements;
}

export default function BlogPostPage({ params }: { params: { slug: string } }) {
  const post = getPostById(params.slug);

  if (!post) {
    notFound();
  }

  const currentUrl = `https://b3d0e50f-4c61-4fac-8435-66faa6872836.canvases.tempo.build/blog/${params.slug}`;

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <div className="bg-gradient-to-br from-primary/5 via-background to-accent/5 border-b border-border">
        <div className="container mx-auto px-4 py-8">
          <Link href="/blog">
            <Button
              variant="ghost"
              className="mb-6 p-0 h-auto font-medium text-primary hover:text-primary/80"
            >
              <ArrowLeft className="w-4 h-4 mr-2" />
              Back to Blog
            </Button>
          </Link>

          <div className="max-w-4xl">
            <div className="flex items-center gap-4 text-sm text-muted-foreground mb-4">
              <span className="bg-primary/10 text-primary px-3 py-1 rounded-full text-xs font-medium">
                {post.category}
              </span>
              <div className="flex items-center gap-1">
                <Calendar className="w-4 h-4" />
                {formatDate(post.date)}
              </div>
              <div className="flex items-center gap-1">
                <Clock className="w-4 h-4" />
                {post.readTime}
              </div>
            </div>

            <h1 className="text-4xl md:text-5xl font-bold text-foreground mb-4">
              {post.title}
            </h1>

            <p className="text-xl text-muted-foreground leading-relaxed mb-6">
              {post.description}
            </p>

            <div className="flex items-center gap-4">
              <SocialShareButtons
                url={currentUrl}
                title={post.title}
                description={post.description}
              />
              <Button variant="outline" size="sm">
                <BookOpen className="w-4 h-4 mr-2" />
                Save for later
              </Button>
            </div>
          </div>
        </div>
      </div>

      {/* Featured Image */}
      <div className="container mx-auto px-4 py-8">
        <div className="max-w-4xl mx-auto">
          <div className="aspect-video overflow-hidden rounded-xl mb-8">
            <img
              src={post.image}
              alt={post.title}
              className="w-full h-full object-cover"
            />
          </div>
        </div>
      </div>

      {/* Content */}
      <div className="container mx-auto px-4 pb-16">
        <div className="max-w-4xl mx-auto">
          <Card className="bg-card">
            <CardHeader className="pb-8">
              <div className="prose prose-lg max-w-none">
                {/* Article content would go here */}
              </div>
            </CardHeader>
            <CardContent className="prose prose-lg max-w-none">
              <div className="space-y-4">
                {renderMarkdownContent(post.content)}
              </div>
            </CardContent>
          </Card>

          {/* Navigation */}
          <div className="mt-12 pt-8 border-t border-border">
            <div className="flex justify-between items-center">
              <Link href="/blog">
                <Button variant="outline">
                  <ArrowLeft className="w-4 h-4 mr-2" />
                  All Posts
                </Button>
              </Link>

              <div className="flex gap-2">
                <SocialShareButtons
                  url={currentUrl}
                  title={post.title}
                  description={post.description}
                />
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
