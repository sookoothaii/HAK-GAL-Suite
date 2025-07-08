import { ScrollArea } from "@/components/ui/scroll-area";
import { useStore } from "@/store/useStore";

export const ProfileView = () => {
  const profileEntity = useStore((state) => state.profileEntity);

  // Mock data for demonstration
  const explicitFacts = [
    "ServerA has IP address 192.168.1.100",
    "ServerA runs Ubuntu 20.04 LTS", 
    "ServerA has 16GB RAM installed"
  ];

  const derivedProperties = [
    "ServerA is likely in production environment",
    "ServerA supports containerized workloads",
    "ServerA has sufficient memory for medium loads"
  ];

  return (
    <div className="h-full space-y-6">
      <div className="flex items-center gap-2 pb-2 border-b border-border/50">
        <div className="w-2 h-2 rounded-full bg-primary animate-neural-pulse"></div>
        <span className="text-sm font-semibold text-primary">Entity Analysis</span>
        <div className="data-chip">
          {profileEntity || "Unknown"}
        </div>
      </div>
      
      <ScrollArea className="h-[calc(100%-4rem)]">
        <div className="space-y-6">
          {/* Explicit Facts */}
          <div className="space-y-3">
            <div className="flex items-center gap-2">
              <h3 className="text-sm font-semibold text-success">Explicit Facts</h3>
              <span className="text-xs bg-success/10 text-success px-2 py-0.5 rounded-full">
                {explicitFacts.length}
              </span>
            </div>
            <div className="space-y-2">
              {explicitFacts.map((fact, index) => (
                <div key={index} className="neural-border rounded-lg p-3 hover:bg-success/5 transition-neural">
                  <div className="flex items-start gap-2">
                    <div className="w-1.5 h-1.5 rounded-full bg-success/60 mt-2 flex-shrink-0"></div>
                    <div className="logic-display flex-1">
                      <span className="text-xs leading-relaxed">{fact}</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Derived Properties */}
          <div className="space-y-3">
            <div className="flex items-center gap-2">
              <h3 className="text-sm font-semibold text-secondary">Derived Properties</h3>
              <span className="text-xs bg-secondary/10 text-secondary px-2 py-0.5 rounded-full">
                {derivedProperties.length}
              </span>
            </div>
            <div className="space-y-2">
              {derivedProperties.map((property, index) => (
                <div key={index} className="neural-border rounded-lg p-3 hover:bg-secondary/5 transition-neural">
                  <div className="flex items-start gap-2">
                    <div className="w-1.5 h-1.5 rounded-full bg-secondary/60 mt-2 flex-shrink-0"></div>
                    <div className="flex-1">
                      <div className="logic-display mb-2">
                        <span className="text-xs leading-relaxed">{property}</span>
                      </div>
                      <div className="text-xs text-muted-foreground">
                        Confidence: {(Math.random() * 0.3 + 0.7).toFixed(2)}
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </ScrollArea>
    </div>
  );
};