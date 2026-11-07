// server/komainu_router.ts
import Groq from "groq-sdk";
import Cerebras from "@cerebras/cerebras_cloud_sdk";

type KomainuRole = "judge" | "commentator";

const GROQ_MODEL = "llama-3.3-70b-versatile"; // 1000回/日
const CEREBRAS_MODEL = "llama3.3-70b"; // 100万トークン/日
const OPENROUTER_MODEL = "meta-llama/llama-3.3-70b-instruct:free"; // 無制限

const groq = process.env.GROQ_API_KEY? new Groq({ apiKey: process.env.GROQ_API_KEY }) : null;
const cerebras = process.env.CEREBRAS_API_KEY? new Cerebras({ apiKey: process.env.CEREBRAS_API_KEY }) : null;

function getSystemPrompt(role: KomainuRole = "judge") {
  if (role === "judge") {
    return `あなたは討論番組「しょぼアプリ」の狛犬先生です。
生徒(3B)より圧倒的に賢い70Bの先生として、論理の穴を厳しく指摘する。
出力は必ず以下のJSONのみ: {"winner": "A or B", "reason": "100字以内の講評", "score_a": 0-100, "score_b": 0-100}`;
  }
  return `あなたは狛犬先生です。討論を面白く解説するコメンテーターです。`;
}

async function callGroq(prompt: string, system: string): Promise<string> {
  if (!groq) throw new Error("GROQ_API_KEY missing");
  const res = await groq.chat.completions.create({
    model: GROQ_MODEL,
    messages: [{ role: "system", content: system }, { role: "user", content: prompt }],
    temperature: 0.3,
    response_format: { type: "json_object" },
  });
  return res.choices[0].message.content!;
}

async function callCerebras(prompt: string, system: string): Promise<string> {
  if (!cerebras) throw new Error("CEREBRAS_API_KEY missing");
  const res = await cerebras.chat.completions.create({
    model: CEREBRAS_MODEL,
    messages: [{ role: "system", content: system }, { role: "user", content: prompt }],
    temperature: 0.3,
  } as any);
  return (res as any).choices[0].message.content;
}

async function callOpenRouter(prompt: string, system: string): Promise<string> {
  if (!process.env.OPENROUTER_API_KEY) throw new Error("OPENROUTER_API_KEY missing");
  const res = await fetch("https://openrouter.ai/api/v1/chat/completions", {
    method: "POST",
    headers: {
      "Authorization": `Bearer ${process.env.OPENROUTER_API_KEY}`,
      "Content-Type": "application/json",
      "HTTP-Referer": "http://localhost:3000",
    },
    body: JSON.stringify({
      model: OPENROUTER_MODEL,
      messages: [{ role: "system", content: system }, { role: "user", content: prompt }],
      temperature: 0.3,
      response_format: { type: "json_object" },
    }),
  });
  if (!res.ok) throw new Error(`OpenRouter error: ${await res.text()}`);
  const data = await res.json();
  return data.choices[0].message.content;
}

// 外から呼ぶのはこれだけ
export async function callKomainu(prompt: string, role: KomainuRole = "judge"): Promise<any> {
  const system = getSystemPrompt(role);
  const ladder = [
    { name: "Groq-70B", fn: callGroq },
    { name: "Cerebras-70B", fn: callCerebras },
    { name: "OpenRouter-70B-free", fn: callOpenRouter },
  ];

  let lastError: any;
  for (const { name, fn } of ladder) {
    try {
      console.log(`[Komainu] Trying ${name}...`);
      const raw = await fn(prompt, system);
      // JSON修復 (3Bで崩れてた対策)
      const json = JSON.parse(raw.match(/\{[\s\S]*\}/)?.[0] || raw);
      console.log(`[Komainu] Success with ${name}`);
      return json;
    } catch (e) {
      console.warn(`[Komainu] ${name} failed:`, e);
      lastError = e;
      continue;
    }
  }
  throw new Error(`All Komainu APIs failed. Last error: ${lastError}`);
}