import { useListJobs } from "@workspace/api-client-react";
import { Link } from "wouter";
import { formatDistanceToNow } from "date-fns";
import { FileUp, Sparkles, Activity, CheckCircle2, XCircle, Clock } from "lucide-react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

export default function Home() {
  const { data: jobs, isLoading } = useListJobs();

  const getStatusIcon = (status: string) => {
    switch (status) {
      case "complete":
        return <CheckCircle2 className="h-4 w-4 text-emerald-500" />;
      case "failed":
        return <XCircle className="h-4 w-4 text-destructive" />;
      case "running":
        return <Activity className="h-4 w-4 text-blue-500 animate-pulse" />;
      default:
        return <Clock className="h-4 w-4 text-muted-foreground" />;
    }
  };

  return (
    <div className="container max-w-6xl py-8 space-y-12">
      <div className="space-y-4">
        <h1 className="text-4xl font-bold tracking-tight">Select Operation Mode</h1>
        <p className="text-lg text-muted-foreground max-w-2xl">
          Choose a method to generate synthetic data. Replicate an existing dataset to match its statistical properties, or describe a new schema from scratch.
        </p>
      </div>

      <div className="grid md:grid-cols-2 gap-6">
        <Link href="/replicate">
          <Card className="h-full hover:border-primary/50 transition-colors cursor-pointer group">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-2xl">
                <FileUp className="h-6 w-6 text-primary" />
                Replicator Mode
              </CardTitle>
              <CardDescription className="text-base">
                Upload a CSV or XLSX file. The engine will profile distributions, correlations, and constraints to generate a synthetic clone.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="h-32 bg-muted/30 rounded-md border border-dashed border-border flex items-center justify-center group-hover:bg-muted/50 transition-colors">
                <span className="text-sm font-mono text-muted-foreground">Drop file to clone</span>
              </div>
            </CardContent>
          </Card>
        </Link>

        <Link href="/create">
          <Card className="h-full hover:border-primary/50 transition-colors cursor-pointer group">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-2xl">
                <Sparkles className="h-6 w-6 text-primary" />
                Creator Mode
              </CardTitle>
              <CardDescription className="text-base">
                Use natural language to describe tables, columns, and relationships. The engine will infer appropriate data types and distributions.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="h-32 bg-muted/30 rounded-md border border-dashed border-border p-4 group-hover:bg-muted/50 transition-colors">
                <p className="text-sm font-mono text-muted-foreground">
                  "Generate a customer table with 10k rows, related to an orders table where each customer has 1-5 orders..."
                </p>
              </div>
            </CardContent>
          </Card>
        </Link>
      </div>

      <div className="space-y-4">
        <h2 className="text-2xl font-semibold tracking-tight">Recent Jobs</h2>
        <Card className="border-border/50">
          <div className="divide-y divide-border/50">
            {isLoading ? (
              <div className="p-8 text-center text-muted-foreground">Loading jobs...</div>
            ) : !jobs || jobs.length === 0 ? (
              <div className="p-8 text-center text-muted-foreground">No recent jobs found.</div>
            ) : (
              jobs.map((job) => (
                <Link key={job.job_id} href={`/jobs/${job.job_id}`}>
                  <div className="flex items-center justify-between p-4 hover:bg-muted/30 transition-colors cursor-pointer">
                    <div className="flex items-center gap-4">
                      {getStatusIcon(job.status)}
                      <div>
                        <div className="font-mono text-sm font-medium">{job.job_id}</div>
                        <div className="text-sm text-muted-foreground flex items-center gap-2">
                          <span className="capitalize">{job.mode}</span>
                          <span>&bull;</span>
                          <span>{formatDistanceToNow(new Date(job.created_at), { addSuffix: true })}</span>
                        </div>
                      </div>
                    </div>
                    <div className="flex items-center gap-4">
                      {job.filename && (
                        <Badge variant="secondary" className="font-mono text-xs">{job.filename}</Badge>
                      )}
                      <Badge variant="outline" className={`capitalize ${
                        job.status === 'complete' ? 'border-emerald-500/30 text-emerald-600 dark:text-emerald-400' :
                        job.status === 'failed' ? 'border-destructive/30 text-destructive' :
                        'border-blue-500/30 text-blue-600 dark:text-blue-400'
                      }`}>
                        {job.status}
                      </Badge>
                    </div>
                  </div>
                </Link>
              ))
            )}
          </div>
        </Card>
      </div>
    </div>
  );
}
