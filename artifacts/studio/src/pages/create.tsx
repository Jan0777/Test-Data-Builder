import { useState } from "react";
import { useLocation } from "wouter";
import { Sparkles, AlertTriangle, ChevronRight, RefreshCw, Play, CheckCircle2, FileSpreadsheet } from "lucide-react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Textarea } from "@/components/ui/textarea";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { useCreateFromQuery, useGenerateFromSpec } from "@workspace/api-client-react";

const EXAMPLES = [
  "500 customers across 3 regions, each with 1-8 orders. Order total correlates with customer tier (bronze/silver/gold). Include product, quantity, and price columns.",
  "5 departments, 200 employees. Salaries skewed lognormal with mean $75k. Employees have hire_date, seniority (junior/mid/senior/lead), and an is_manager boolean.",
  "A transactions table with 10,000 rows: amount (lognormal), timestamp, category (groceries/dining/transport/entertainment/other), merchant_name, and a status column.",
];

type Step = "input" | "loading" | "review" | "generating";

export default function Create() {
  const [, navigate] = useLocation();
  const [step, setStep] = useState<Step>("input");
  const [query, setQuery] = useState("");
  const [preview, setPreview] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const [conflicts, setConflicts] = useState<string[]>([]);

  const createMutation = useCreateFromQuery();
  const generateMutation = useGenerateFromSpec();

  const handleSubmit = () => {
    if (!query.trim()) return;
    setError(null);
    setConflicts([]);
    setStep("loading");
    createMutation.mutate(
      { data: { query } },
      {
        onSuccess: (data: any) => {
          setPreview(data);
          setConflicts(data.conflicts || []);
          setStep("review");
        },
        onError: (err: any) => {
          const detail = err?.data;
          setError(detail?.error || "Failed to parse query");
          setConflicts(detail?.conflicts || []);
          setStep("input");
        },
      }
    );
  };

  const handleGenerate = () => {
    if (!preview?.spec) return;
    setStep("generating");
    generateMutation.mutate(
      { data: { spec: preview.spec, format: "csv" } },
      {
        onSuccess: (data: any) => navigate(`/jobs/${data.job_id}`),
        onError: (err: any) => {
          setError(err?.data?.error || "Generation failed");
          setStep("review");
        },
      }
    );
  };

  return (
    <div className="container max-w-4xl py-8 space-y-6">
      <div className="space-y-1">
        <h1 className="text-3xl font-bold tracking-tight">Creator</h1>
        <p className="text-muted-foreground">
          Describe the data you need in plain language. The engine will infer types, distributions, and relationships.
        </p>
      </div>

      <div className="flex items-center gap-2 text-sm">
        {(["input", "review"] as Step[]).map((s, i) => (
          <span key={s} className="flex items-center gap-2">
            {i > 0 && <ChevronRight className="h-3 w-3 text-muted-foreground" />}
            <span className={`font-medium ${step === s || (step === "generating" && s === "review") || (step === "loading" && s === "input") ? "text-primary" : "text-muted-foreground"}`}>
              {i + 1}. {s === "input" ? "Describe" : "Confirm & Generate"}
            </span>
          </span>
        ))}
      </div>

      {error && (
        <Alert variant="destructive">
          <AlertTriangle className="h-4 w-4" />
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {(step === "input" || step === "loading") && (
        <div className="space-y-4">
          <Card>
            <CardContent className="pt-6 space-y-4">
              <Textarea
                placeholder="Describe the tables you want to generate. Be specific about row counts, column names, data types, and any relationships between tables."
                className="min-h-[160px] font-mono text-sm resize-y"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                disabled={step === "loading"}
              />
              <div className="flex items-center justify-between">
                <p className="text-xs text-muted-foreground">Powered by Claude</p>
                <Button onClick={handleSubmit} disabled={!query.trim() || step === "loading"}>
                  {step === "loading" ? (
                    <><RefreshCw className="h-4 w-4 mr-2 animate-spin" />Interpreting...</>
                  ) : (
                    <><Sparkles className="h-4 w-4 mr-2" />Interpret Query</>
                  )}
                </Button>
              </div>
            </CardContent>
          </Card>

          <div className="space-y-2">
            <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider">Examples</p>
            <div className="space-y-2">
              {EXAMPLES.map((ex, i) => (
                <button
                  key={i}
                  className="w-full text-left p-3 rounded-md border border-border/50 hover:border-primary/40 hover:bg-muted/30 transition-colors text-sm font-mono text-muted-foreground"
                  onClick={() => setQuery(ex)}
                >
                  {ex}
                </button>
              ))}
            </div>
          </div>
        </div>
      )}

      {(step === "review" || step === "generating") && preview && (
        <div className="space-y-4">
          <Alert>
            <CheckCircle2 className="h-4 w-4 text-emerald-500" />
            <AlertDescription>{preview.summary}</AlertDescription>
          </Alert>

          {conflicts.length > 0 && (
            <Alert>
              <AlertTriangle className="h-4 w-4" />
              <AlertDescription>
                <ul className="list-disc list-inside space-y-0.5">
                  {conflicts.map((c: string, i: number) => <li key={i}>{c}</li>)}
                </ul>
              </AlertDescription>
            </Alert>
          )}

          <Card>
            <CardHeader className="pb-3">
              <div className="flex items-center justify-between">
                <CardTitle className="text-lg">Interpreted Spec</CardTitle>
                <div className="flex gap-2">
                  <Badge variant="outline">{preview.spec?.tables?.length ?? 0} table(s)</Badge>
                  <Badge variant="outline">{preview.spec?.relationships?.length ?? 0} relationship(s)</Badge>
                </div>
              </div>
              <CardDescription>Review the interpreted schema before generating</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {preview.spec?.tables?.map((table: any) => (
                <div key={table.name} className="space-y-2">
                  <div className="flex items-center gap-2">
                    <FileSpreadsheet className="h-4 w-4 text-primary" />
                    <span className="font-mono font-semibold">{table.name}</span>
                    <Badge variant="secondary" className="text-xs">{table.row_count} rows</Badge>
                    {table.primary_key && (
                      <Badge variant="outline" className="text-xs font-mono">PK: {table.primary_key}</Badge>
                    )}
                  </div>
                  <div className="ml-6 grid grid-cols-2 md:grid-cols-3 gap-1.5">
                    {table.columns?.map((col: any) => (
                      <div key={col.name} className="flex items-center gap-1.5 text-xs p-1.5 rounded bg-muted/40 border border-border/50">
                        <span className="font-mono truncate">{col.name}</span>
                        <Badge className="text-[10px] px-1 py-0 shrink-0" variant="outline">{col.type}</Badge>
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </CardContent>
          </Card>

          <div className="flex items-center justify-between">
            <Button variant="outline" onClick={() => setStep("input")} disabled={step === "generating"}>
              Edit Query
            </Button>
            <Button size="lg" onClick={handleGenerate} disabled={step === "generating"}>
              {step === "generating" ? (
                <><RefreshCw className="h-4 w-4 mr-2 animate-spin" />Generating...</>
              ) : (
                <><Play className="h-4 w-4 mr-2" />Generate Synthetic Data</>
              )}
            </Button>
          </div>
        </div>
      )}
    </div>
  );
}
