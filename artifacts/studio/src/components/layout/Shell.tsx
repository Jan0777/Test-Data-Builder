import { ReactNode } from "react";
import { Link, useLocation } from "wouter";
import { Database, FileUp, Sparkles, LayoutDashboard, Settings } from "lucide-react";
import { useTheme } from "@/components/theme-provider";
import { Button } from "@/components/ui/button";

export default function Shell({ children }: { children: ReactNode }) {
  const [location] = useLocation();
  const { theme, setTheme } = useTheme();

  return (
    <div className="min-h-screen flex flex-col bg-background">
      <header className="sticky top-0 z-50 w-full border-b border-border/50 bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
        <div className="container flex h-14 items-center px-4">
          <div className="mr-4 flex">
            <Link href="/" className="mr-6 flex items-center space-x-2">
              <Database className="h-5 w-5 text-primary" />
              <span className="hidden font-bold sm:inline-block font-mono tracking-tight">
                SYNTHETIC_DATA_STUDIO
              </span>
            </Link>
            <nav className="flex items-center space-x-6 text-sm font-medium">
              <Link
                href="/"
                className={`transition-colors hover:text-foreground/80 ${
                  location === "/" ? "text-foreground" : "text-foreground/60"
                }`}
              >
                Dashboard
              </Link>
              <Link
                href="/replicate"
                className={`transition-colors hover:text-foreground/80 ${
                  location.startsWith("/replicate") ? "text-foreground" : "text-foreground/60"
                }`}
              >
                Replicator
              </Link>
              <Link
                href="/create"
                className={`transition-colors hover:text-foreground/80 ${
                  location.startsWith("/create") ? "text-foreground" : "text-foreground/60"
                }`}
              >
                Creator
              </Link>
            </nav>
          </div>
          <div className="ml-auto flex items-center space-x-4">
            <Button
              variant="ghost"
              size="icon"
              onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
            >
              <Settings className="h-4 w-4" />
              <span className="sr-only">Toggle theme</span>
            </Button>
          </div>
        </div>
      </header>
      <main className="flex-1 flex flex-col">{children}</main>
    </div>
  );
}
