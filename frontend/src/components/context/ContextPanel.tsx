import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Search, User } from "lucide-react";
import { useStore } from "@/store/useStore";
import { ProfileView } from "./ProfileView";
import { RagView } from "./RagView";

export const ContextPanel = () => {
  const { rightPanelView, profileEntity } = useStore();

  const getTitle = () => {
    return rightPanelView === "rag" ? "RAG Context" : `Profile: ${profileEntity}`;
  };

  return (
    <Card className="flex flex-col neural-border shadow-card transition-neural hover:shadow-neural">
      <CardHeader className="pb-4">
        <CardTitle className="flex items-center gap-2 text-lg">
          <div className="w-6 h-6 rounded bg-gradient-data flex items-center justify-center">
            {rightPanelView === "rag" ? (
              <Search className="w-3 h-3 text-primary-foreground" />
            ) : (
              <User className="w-3 h-3 text-primary-foreground" />
            )}
          </div>
          {getTitle()}
        </CardTitle>
      </CardHeader>
      <CardContent className="flex-1">
        {rightPanelView === "rag" ? <RagView /> : <ProfileView />}
      </CardContent>
    </Card>
  );
};