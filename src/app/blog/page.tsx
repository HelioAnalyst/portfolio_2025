import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Calendar, Clock, ArrowRight } from "lucide-react";
import Link from "next/link";
import { getAllPosts, getFeaturedPosts } from "@/lib/blog";

// Type the post from your data source shape
type BlogPost = ReturnType<typeof getAllPosts>[number];

function formatDate(dateString: string) {
  return new Date(dateString).toLocaleDateString("en-GB", {
    year: "numeric",
    month: "long",
    day: "numeric",
  });
}

function BlogPostCard({
  post,
  featured = false,
}: {
  post: BlogPost;
  featured?: boolean;
}) {
  return (
    <Card
      className={`group hover:shadow-lg transition-all duration-300 bg-card ${
        featured ? "md:col-span-2 lg:col-span-3" : ""
      }`}
    >
      <div className={featured ? "md:flex" : ""}>
        <div className={featured ? "md:w-1/2" : ""}>
          <div
            className={`aspect-video overflow-hidden rounded-t-xl ${
              featured ? "md:rounded-l-xl md:rounded-tr-none" : ""
            }`}
          >
            <img
              src={post.image}
              alt={post.title}
              className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
              loading="lazy"
            />
          </div>
        </div>
        <div className={featured ? "md:w-1/2" : ""}>
          <CardHeader className={featured ? "pb-4" : ""}>
            <div className="flex flex-wrap items-center gap-4 text-sm text-muted-foreground mb-2">
              <span className="bg-primary/10 text-primary px-2 py-1 rounded-full text-xs font-medium">
                {post.category}
              </span>
              <div className="flex items-center gap-1">
                <Calendar className="w-3 h-3" />
                {formatDate(post.date)}
              </div>
              <div className="flex items-center gap-1">
                <Clock className="w-3 h-3" />
                {post.readTime}
              </div>
            </div>
            <CardTitle
              className={`group-hover:text-primary transition-colors ${
                featured ? "text-2xl" : "text-xl"
              }`}
            >
              {post.title}
            </CardTitle>
            <CardDescription className={featured ? "text-base" : ""}>
              {post.description}
            </CardDescription>
          </CardHeader>
          <CardContent className="pt-0">
            <Button
              asChild
              variant="ghost"
              className="group/btn p-0 h-auto font-medium text-primary hover:text-primary/80"
            >
              <Link href={`/blog/${post.id}`}>
                Read more
                <ArrowRight className="w-4 h-4 ml-1 group-hover/btn:translate-x-1 transition-transform" />
              </Link>
            </Button>
          </CardContent>
        </div>
      </div>
    </Card>
  );
}

export default function BlogPage() {
  const allPosts = getAllPosts();
  const featuredPosts = getFeaturedPosts();
  const featuredPost = featuredPosts[0];
  const regularPosts = allPosts.filter((post) => !post.featured);

  // ---- No-Set category dedupe (compatible with older TS targets) ----
  const categoryMap: Record<string, true> = {};
  for (const p of allPosts) categoryMap[p.category] = true;
  const categories = Object.keys(categoryMap).sort();
  // -------------------------------------------------------------------

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <div className="bg-gradient-to-br from-primary/5 via-background to-accent/5 border-b border-border">
        <div className="container mx-auto px-4 py-16">
          <div className="max-w-3xl">
            <h1 className="text-4xl md:text-5xl font-bold text-foreground mb-4">
              Technical Blog
            </h1>
            <p className="text-xl text-muted-foreground leading-relaxed">
              Insights, tutorials, and best practices in Python development,
              data analysis, and machine learning. Sharing knowledge from
              real-world projects and experiences.
            </p>
          </div>
        </div>
      </div>

      {/* Content */}
      <div className="container mx-auto px-4 py-12">
        {/* Featured Post */}
        {featuredPost && (
          <div className="mb-12">
            <h2 className="text-2xl font-semibold text-foreground mb-6">
              Featured Post
            </h2>
            <div className="grid grid-cols-1">
              <BlogPostCard post={featuredPost} featured />
            </div>
          </div>
        )}

        {/* Regular Posts */}
        <div className="mb-8">
          <h2 className="text-2xl font-semibold text-foreground mb-6">
            Latest Posts
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {regularPosts.map((post) => (
              <BlogPostCard key={post.id} post={post} />
            ))}
          </div>
        </div>

        {/* Categories */}
        <div className="mt-16">
          <h2 className="text-2xl font-semibold text-foreground mb-6">
            Categories
          </h2>
          <div className="flex flex-wrap gap-3">
            {categories.map((category) => (
              <Button key={category} variant="outline" className="rounded-full">
                {category}
              </Button>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
