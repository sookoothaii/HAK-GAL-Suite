// react_state_debug.js - Erweiterte Frontend-Debugging-Tools
// Kopieren Sie das in die Browser-Konsole (F12)

// 1. React State Inspector
function inspectReactState() {
    console.log("🔍 React State Inspector...");
    
    // Finde React Root
    const reactRoot = document.querySelector('#root');
    if (!reactRoot) {
        console.error("❌ React Root nicht gefunden!");
        return;
    }
    
    // Versuche React DevTools Zugriff
    const reactInstance = reactRoot._reactInternalInstance || 
                         reactRoot._reactInternalFiber || 
                         Object.keys(reactRoot).find(key => key.startsWith("__reactInternalInstance"));
    
    console.log("⚛️ React Instance:", !!reactInstance);
    
    // Prüfe ob Messages im DOM sind
    const messageElements = document.querySelectorAll('[class*="space-y-4"] > div');
    console.log("💬 Message-Elemente im DOM:", messageElements.length);
    
    messageElements.forEach((el, index) => {
        const hasUserIcon = el.querySelector('span')?.textContent === '👤';
        const hasSystemIcon = el.querySelector('span')?.textContent === '🤖';
        const content = el.querySelector('pre')?.textContent || "Kein Content";
        
        console.log(`   Message ${index}:`, {
            type: hasUserIcon ? 'user' : hasSystemIcon ? 'system' : 'unknown',
            content: content.substring(0, 50) + (content.length > 50 ? '...' : '')
        });
    });
    
    // Prüfe Knowledge Base Elemente
    const knowledgeItems = document.querySelectorAll('[class*="space-y-2"] > div[class*="bg-gray-800"]');
    console.log("🧠 Knowledge Base Items:", knowledgeItems.length);
    
    return {
        messagesInDOM: messageElements.length,
        knowledgeItemsInDOM: knowledgeItems.length
    };
}

// 2. Live State Monitor
function startStateMonitor() {
    console.log("📊 Starte Live State Monitor...");
    
    let lastMessageCount = 0;
    let lastKnowledgeCount = 0;
    
    const monitor = setInterval(() => {
        const messageElements = document.querySelectorAll('[class*="space-y-4"] > div');
        const knowledgeItems = document.querySelectorAll('[class*="space-y-2"] > div[class*="bg-gray-800"]');
        
        if (messageElements.length !== lastMessageCount) {
            console.log(`🔄 Messages geändert: ${lastMessageCount} → ${messageElements.length}`);
            lastMessageCount = messageElements.length;
        }
        
        if (knowledgeItems.length !== lastKnowledgeCount) {
            console.log(`🔄 Knowledge geändert: ${lastKnowledgeCount} → ${knowledgeItems.length}`);
            lastKnowledgeCount = knowledgeItems.length;
        }
    }, 1000);
    
    console.log("✅ Monitor gestartet. Zum Stoppen: clearInterval(monitor)");
    return monitor;
}

// 3. Simuliere Frontend-Command
async function simulateFrontendCommand(command = "status") {
    console.log(`🎭 Simuliere Frontend-Command: ${command}`);
    
    // Zähle aktuelle Messages
    const beforeMessages = document.querySelectorAll('[class*="space-y-4"] > div').length;
    console.log("📊 Messages vor Command:", beforeMessages);
    
    // Sende Input ins Frontend
    const inputField = document.querySelector('input[placeholder*="command"]');
    if (inputField) {
        // Simuliere User-Input
        inputField.value = command;
        inputField.dispatchEvent(new Event('change', { bubbles: true }));
        
        // Finde und klicke Send-Button
        const sendButton = inputField.parentElement.querySelector('button[class*="bg-blue-600"]');
        if (sendButton) {
            console.log("📤 Sende Command über Frontend...");
            sendButton.click();
            
            // Warte und prüfe Ergebnis
            setTimeout(() => {
                const afterMessages = document.querySelectorAll('[class*="space-y-4"] > div').length;
                console.log("📊 Messages nach Command:", afterMessages);
                console.log("📈 Differenz:", afterMessages - beforeMessages);
                
                if (afterMessages > beforeMessages) {
                    console.log("✅ Command erfolgreich - Messages hinzugefügt!");
                    inspectReactState();
                } else {
                    console.log("❌ Command scheint nicht anzukommen - keine neuen Messages");
                }
            }, 2000);
        } else {
            console.error("❌ Send-Button nicht gefunden!");
        }
    } else {
        console.error("❌ Input-Field nicht gefunden!");
    }
}

// 4. Message-Flow Tracer
async function traceMessageFlow(command = "status") {
    console.log("🔍 Trace Message Flow für:", command);
    
    // Step 1: Vor dem Command
    const beforeState = inspectReactState();
    console.log("🔹 State vor Command:", beforeState);
    
    // Step 2: API Call simulieren (parallel zum Frontend)
    console.log("🔹 API Call...");
    try {
        const response = await fetch("http://localhost:5001/api/command", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ command: command })
        });
        
        const data = await response.json();
        console.log("🔹 API Response erhalten:", {
            status: data.status,
            hasChat: !!data.chatResponse,
            chatLength: data.chatResponse?.length || 0,
            knowledge: data.permanentKnowledge?.length || 0
        });
        
        // Step 3: Frontend Command parallel
        simulateFrontendCommand(command);
        
    } catch (error) {
        console.error("🔹 API Call failed:", error);
    }
}

// 5. Component-Tree Inspector
function inspectComponentTree() {
    console.log("🌳 Component Tree Inspector...");
    
    // Hauptcontainer finden
    const mainContainer = document.querySelector('[class*="grid-cols-1"]');
    if (!mainContainer) {
        console.error("❌ Main Container nicht gefunden!");
        return;
    }
    
    // Cards zählen
    const cards = mainContainer.querySelectorAll('[class*="bg-gray-800/50"]');
    console.log("📱 Cards gefunden:", cards.length);
    
    cards.forEach((card, index) => {
        const title = card.querySelector('[class*="text-gray-100"]')?.textContent;
        const content = card.querySelector('[class*="flex-1"]');
        
        console.log(`   Card ${index}: ${title}`);
        
        if (title === "Interaction Panel") {
            const messages = card.querySelectorAll('[class*="space-y-4"] > div');
            console.log(`     💬 Messages: ${messages.length}`);
        } else if (title === "Knowledge Base") {
            const knowledge = card.querySelectorAll('[class*="space-y-2"] > div[class*="bg-gray-800"]');
            console.log(`     🧠 Knowledge: ${knowledge.length}`);
        }
    });
    
    return { cards: cards.length };
}

// Auto-Start
console.log("🔧 React State Debug Tools geladen!");
console.log("🚀 Verfügbare Funktionen:");
console.log("   - inspectReactState()");
console.log("   - startStateMonitor()"); 
console.log("   - simulateFrontendCommand('status')");
console.log("   - traceMessageFlow('status')");
console.log("   - inspectComponentTree()");
console.log("");
console.log("💡 Empfehlung: Führen Sie traceMessageFlow('status') aus!");

// Sofortiger Check
inspectComponentTree();
