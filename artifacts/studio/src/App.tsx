import { Switch, Route, Router as WouterRouter } from "wouter";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { Toaster } from "@/components/ui/toaster";
import { TooltipProvider } from "@/components/ui/tooltip";
import { ThemeProvider } from "@/components/theme-provider";
import NotFound from "@/pages/not-found";
import Home from "@/pages/home";
import Replicate from "@/pages/replicate";
import Create from "@/pages/create";
import JobResults from "@/pages/job-results";
import Shell from "@/components/layout/Shell";

const queryClient = new QueryClient();

function Router() {
  return (
    <Shell>
      <Switch>
        <Route path="/" component={Home} />
        <Route path="/replicate" component={Replicate} />
        <Route path="/create" component={Create} />
        <Route path="/jobs/:jobId" component={JobResults} />
        <Route component={NotFound} />
      </Switch>
    </Shell>
  );
}

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <ThemeProvider defaultTheme="dark" storageKey="sds-theme">
        <TooltipProvider>
          <WouterRouter base={import.meta.env.BASE_URL.replace(/\/$/, "")}>
            <Router />
          </WouterRouter>
          <Toaster />
        </TooltipProvider>
      </ThemeProvider>
    </QueryClientProvider>
  );
}

export default App;
