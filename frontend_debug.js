// frontend_debug.js - Debug-Tools für HAK-GAL Frontend
// Öffnen Sie die Browser-Konsole (F12) und fügen Sie diese Funktionen ein

// 1. Backend-Connection Test
async function testBackendConnection() {
    console.log("🔍 Teste Backend-Verbindung...");
    
    try {
        const response = await fetch("http://localhost:5001/api/test", {
            method: "GET",
            headers: { "Content-Type": "application/json" }
        });
        
        if (response.ok) {
            const data = await response.json();
            console.log("✅ Backend erreichbar:", data);
            return true;
        } else {
            console.error("❌ Backend Response-Fehler:", response.status, response.statusText);
            return false;
        }
    } catch (error) {
        console.error("❌ Backend nicht erreichbar:", error.message);
        console.error("❌ Mögliche Ursachen:");
        console.error("   - Backend nicht gestartet");
        console.error("   - CORS-Problem");
        console.error("   - Falsche URL/Port");
        return false;
    }
}

// 2. API-Command Test
async function testApiCommand(command = "status") {
    console.log(`🔍 Teste API-Command: ${command}`);
    
    try {
        const response = await fetch("http://localhost:5001/api/command", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ command: command })
        });
        
        console.log("📨 Response Status:", response.status);
        console.log("📨 Response Headers:", Object.fromEntries(response.headers.entries()));
        
        if (response.ok) {
            const data = await response.json();
            console.log("✅ API Response:", data);
            
            // Analysiere die Response-Struktur
            console.log("📊 Response-Analyse:");
            console.log("   - status:", data.status);
            console.log("   - lastCommand:", data.lastCommand);
            console.log("   - chatResponse:", data.chatResponse ? "✅ vorhanden" : "❌ fehlt");
            console.log("   - permanentKnowledge:", data.permanentKnowledge ? data.permanentKnowledge.length : "❌ fehlt");
            console.log("   - learningSuggestions:", data.learningSuggestions ? data.learningSuggestions.length : "❌ fehlt");
            
            return data;
        } else {
            const errorData = await response.text();
            console.error("❌ API Error Response:", errorData);
            return null;
        }
    } catch (error) {
        console.error("❌ API Request failed:", error.message);
        return null;
    }
}

// 3. Frontend State Inspector
function inspectFrontendState() {
    console.log("🔍 Frontend State Check:");
    
    // Prüfe aktuelle URL
    console.log("📍 Current URL:", window.location.href);
    
    // Prüfe ob React App läuft
    const reactRoot = document.querySelector('#root');
    console.log("⚛️ React Root gefunden:", reactRoot ? "✅" : "❌");
    
    // Prüfe Network Requests im DevTools
    console.log("🌐 Network-Tab öffnen für Request-Analyse");
    
    // Prüfe Console Errors
    console.log("⚠️ Console-Errors: Schauen Sie nach roten Fehlermeldungen oben");
    
    return {
        url: window.location.href,
        reactRoot: !!reactRoot,
        timestamp: new Date().toISOString()
    };
}

// 4. Vollständiger Diagnose-Test
async function fullDiagnosis() {
    console.log("🚀 Starte vollständige Diagnose...");
    console.log("=" * 50);
    
    // Step 1: Frontend Check
    console.log("SCHRITT 1: Frontend-Status");
    const frontendState = inspectFrontendState();
    
    // Step 2: Backend Connection
    console.log("\nSCHRITT 2: Backend-Verbindung");
    const backendOnline = await testBackendConnection();
    
    if (!backendOnline) {
        console.error("❌ Backend nicht erreichbar - stoppe Diagnose");
        return false;
    }
    
    // Step 3: API Commands
    console.log("\nSCHRITT 3: API-Command Tests");
    await testApiCommand("status");
    await testApiCommand("show");
    await testApiCommand("ask test");
    
    // Step 4: Zusammenfassung
    console.log("\nZUSAMMENFASSUNG:");
    console.log("Frontend:", frontendState.reactRoot ? "✅ OK" : "❌ Problem");
    console.log("Backend:", backendOnline ? "✅ OK" : "❌ Problem");
    
    console.log("\n🔧 Nächste Schritte:");
    console.log("1. Prüfen Sie das Backend-Terminal auf Fehlermeldungen");
    console.log("2. Schauen Sie im Network-Tab nach fehlgeschlagenen Requests");
    console.log("3. Testen Sie eine einfache Nachricht im Frontend");
    
    return true;
}

// 5. CORS-Test
async function testCORS() {
    console.log("🔒 CORS-Test...");
    
    try {
        // OPTIONS Preflight Request
        const optionsResponse = await fetch("http://localhost:5001/api/command", {
            method: "OPTIONS",
            headers: {
                "Origin": window.location.origin,
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "Content-Type"
            }
        });
        
        console.log("✅ CORS Preflight:", optionsResponse.status);
        
        // Actual POST Request
        const postResponse = await fetch("http://localhost:5001/api/command", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ command: "status" })
        });
        
        console.log("✅ CORS POST Request:", postResponse.status);
        
        return true;
    } catch (error) {
        console.error("❌ CORS-Fehler:", error.message);
        return false;
    }
}

// Automatisch starten
console.log("🔧 Frontend Debug-Tools geladen!");
console.log("🚀 Verfügbare Funktionen:");
console.log("   - testBackendConnection()");
console.log("   - testApiCommand('ihr_command')");
console.log("   - inspectFrontendState()");
console.log("   - fullDiagnosis()");
console.log("   - testCORS()");
console.log("");
console.log("💡 Tipp: Führen Sie fullDiagnosis() für einen kompletten Check aus!");
