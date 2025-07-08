import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Button } from "@/components/ui/button";
import { Upload, Database, Brain, FileText } from "lucide-react";
import { useStore } from "@/store/useStore";
import { KnowledgeItem } from "./KnowledgeItem";
import { LearningSuggestionItem } from "./LearningSuggestionItem";
import { DataSourceItem } from "./DataSourceItem";

export const KnowledgePanel = () => {
  const { permanentKnowledge, learningSuggestions, dataSources } = useStore();

  return (
    <Card className="flex flex-col neural-border shadow-card transition-neural hover:shadow-neural">
      <CardHeader className="pb-4">
        <CardTitle className="flex items-center gap-2 text-lg">
          <div className="w-6 h-6 rounded bg-gradient-neural flex items-center justify-center">
            <Database className="w-3 h-3 text-primary-foreground" />
          </div>
          Knowledge Base
        </CardTitle>
      </CardHeader>
      <CardContent className="flex-1 space-y-6">
        {/* Permanent Knowledge */}
        <div className="space-y-3">
          <div className="flex items-center gap-2">
            <h3 className="text-sm font-semibold text-primary">Permanent Knowledge Base</h3>
            <span className="text-xs bg-primary/10 text-primary px-2 py-0.5 rounded-full">
              {permanentKnowledge.length}
            </span>
          </div>
          <ScrollArea className="h-36 neural-border rounded-lg p-3">
            {permanentKnowledge.length === 0 ? (
              <div className="flex flex-col items-center justify-center h-full text-center">
                <Database className="w-8 h-8 text-muted-foreground/50 mb-2" />
                <p className="text-xs text-muted-foreground">No permanent knowledge yet.</p>
                <p className="text-xs text-muted-foreground/70">Facts will appear here once learned.</p>
              </div>
            ) : (
              <div className="space-y-2">
                {permanentKnowledge.map((item) => (
                  <KnowledgeItem key={item.id} item={item} />
                ))}
              </div>
            )}
          </ScrollArea>
        </div>

        {/* Learning Suggestions */}
        <div className="space-y-3">
          <div className="flex items-center gap-2">
            <h3 className="text-sm font-semibold text-secondary">Learning Suggestions</h3>
            <span className="text-xs bg-secondary/10 text-secondary px-2 py-0.5 rounded-full">
              {learningSuggestions.length}
            </span>
          </div>
          <ScrollArea className="h-36 neural-border rounded-lg p-3">
            {learningSuggestions.length === 0 ? (
              <div className="flex flex-col items-center justify-center h-full text-center">
                <Brain className="w-8 h-8 text-muted-foreground/50 mb-2" />
                <p className="text-xs text-muted-foreground">No learning suggestions available.</p>
                <p className="text-xs text-muted-foreground/70">AI-generated insights will appear here.</p>
              </div>
            ) : (
              <div className="space-y-2">
                {learningSuggestions.map((item) => (
                  <LearningSuggestionItem key={item.id} item={item} />
                ))}
              </div>
            )}
          </ScrollArea>
        </div>

        {/* Data Sources */}
        <div className="space-y-3">
          <div className="flex items-center gap-2">
            <h3 className="text-sm font-semibold text-accent">Data Sources</h3>
            <span className="text-xs bg-accent/10 text-accent px-2 py-0.5 rounded-full">
              {dataSources.length}
            </span>
          </div>
          <ScrollArea className="h-28 neural-border rounded-lg p-3 mb-3">
            {dataSources.length === 0 ? (
              <div className="flex flex-col items-center justify-center h-full text-center">
                <FileText className="w-6 h-6 text-muted-foreground/50 mb-1" />
                <p className="text-xs text-muted-foreground">No data sources loaded.</p>
              </div>
            ) : (
              <div className="space-y-1">
                {dataSources.map((source) => (
                  <DataSourceItem key={source.id} source={source} />
                ))}
              </div>
            )}
          </ScrollArea>
          <Button className="w-full bg-gradient-data hover:opacity-90 transition-neural" size="sm">
            <Upload className="w-4 h-4 mr-2" />
            Upload Document
          </Button>
        </div>
      </CardContent>
    </Card>
  );
};