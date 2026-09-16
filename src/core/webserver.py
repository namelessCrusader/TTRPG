"""Local browser UI — SPATIAL: click a tile, act on that tile (or type to an NPC).

Instead of a flat action list (which explodes as verb×item×direction), the map
is a clickable grid: selecting a cell shows only the actions for that cell, plus
a free-text box when it holds an NPC. Stdlib http.server only; drives the live
engine and, with --brain torch / --voice reyna, the local GPU models.

    python -m src.core web --brain torch --voice reyna   # open http://127.0.0.1:8000
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import engine, social
from .seed import guildhall as scene

PAGE = """<!doctype html><html><head><meta charset=utf-8><title>Mark-1</title>
<style>
 body{background:#14161a;color:#d7dae0;font:14px/1.5 ui-monospace,Menlo,Consolas,monospace;margin:0;padding:16px}
 h1{font-size:15px;color:#8ab4f8;margin:0 0 12px;letter-spacing:.05em}
 .wrap{display:flex;gap:20px;flex-wrap:wrap}
 .col{flex:1;min-width:320px}
 #grid{display:inline-block;background:#0d0f12;border:1px solid #262b33;border-radius:8px;padding:10px;user-select:none}
 .row{display:flex}
 .cell{width:26px;height:26px;line-height:26px;text-align:center;font-size:17px;cursor:pointer;border:1px solid transparent;border-radius:4px}
 .cell:hover{background:#1c2431}
 .cell.sel{border-color:#8ab4f8;background:#22304a}
 .cell.player{color:#8ab4f8;font-weight:bold}
 #info{margin:10px 0;color:#9fb0c3;min-height:20px}
 h2{font-size:12px;color:#6b7280;text-transform:uppercase;letter-spacing:.08em;margin:14px 0 4px}
 #general,#at{display:flex;flex-wrap:wrap;gap:6px}
 button{background:#1e2733;color:#cfe0ff;border:1px solid #33465c;border-radius:6px;padding:6px 10px;cursor:pointer;font:inherit}
 button:hover{background:#28374a}
 button.new{background:#3a2330;border-color:#6a3c4e;color:#ffd7e0;padding:3px 9px}
 #saybox{display:none;gap:6px;margin:10px 0;align-items:center}
 input#utter{flex:1;min-width:180px;background:#0d0f12;color:#d7dae0;border:1px solid #33465c;border-radius:6px;padding:6px;font:inherit}
 #log{background:#0d0f12;border:1px solid #262b33;border-radius:8px;padding:12px;height:240px;overflow:auto;white-space:pre-wrap;margin-top:12px}
 .hint{color:#6b7280;font-size:12px;margin-top:6px}
</style></head><body>
<h1>MARK-1 · the gilded cinder <button class=new onclick=doNew()>restart</button></h1>
<div class=wrap>
 <div class=col>
   <div id=levels style="margin-bottom:6px"></div>
   <div id=grid></div>
   <div id=info>click a tile to inspect and act on it</div>
   <div class=hint>@ you · e npc · * fire · O barrel · ! prize · o oil · a acid · ~ water · # wood · x dead</div>
 </div>
 <div class=col>
   <h2>do anything</h2>
   <div style="display:flex;gap:6px;margin-bottom:4px">
     <input id=doit placeholder="pour the oil under the beam and light it…" style="flex:1;min-width:180px;background:#0d0f12;color:#d7dae0;border:1px solid #33465c;border-radius:6px;padding:6px;font:inherit">
     <button onclick=doAct()>act</button></div>
   <h2>general</h2><div id=general></div>
   <h2 id=athdr>at selected tile</h2><div id=at></div>
   <div id=saybox><span id=sayto></span>
     <select id=tone style="background:#0d0f12;color:#d7dae0;border:1px solid #33465c;border-radius:6px;padding:6px;font:inherit">
       <option value="">neutral</option><option value="friendly">friendly</option>
       <option value="threatening">threatening</option><option value="insulting">mocking</option>
     </select>
     <input id=utter placeholder="type anything…"><button onclick=doSay()>send</button></div>
   <div id=log></div>
 </div>
</div>
<script>
let state=null, sel=null, acc=[], curZ=null;
function btn(o){const b=document.createElement('button');b.textContent=o.label;b.onclick=()=>doStep(o.i);return b;}
function render(){
 const lv=document.getElementById('levels'); lv.innerHTML='';
 for(let i=state.zmax-1;i>=0;i--){const b=document.createElement('button');
   b.textContent=(i===state.z?'▸ ':'')+'level '+i+(i===state.player[2]?' (you)':'');
   b.onclick=()=>{curZ=i;refreshZ();};lv.appendChild(b);}
 const g=document.getElementById('grid'); g.innerHTML='';
 state.grid.forEach((row,y)=>{const r=document.createElement('div');r.className='row';
   row.forEach((cell,x)=>{const c=document.createElement('span');c.className='cell';c.textContent=cell.ch;
     if(cell.dim)c.style.opacity=0.35;
     if(x==state.player[0]&&y==state.player[1]&&state.z==state.player[2])c.classList.add('player');
     if(sel&&sel[0]==x&&sel[1]==y)c.classList.add('sel');
     c.onclick=()=>{sel=[x,y];render();};r.appendChild(c);});
   g.appendChild(r);});
 document.getElementById('info').textContent = sel? state.cells[sel[0]+','+sel[1]] : 'click a tile to inspect and act on it';
 const gen=document.getElementById('general');gen.innerHTML='';
 state.options.filter(o=>!o.target).forEach(o=>gen.appendChild(btn(o)));
 const at=document.getElementById('at');at.innerHTML='';
 let n=0; if(sel){state.options.filter(o=>o.target&&o.target[0]==sel[0]&&o.target[1]==sel[1]).forEach(o=>{at.appendChild(btn(o));n++;});}
 document.getElementById('athdr').textContent = sel? ('at ('+sel[0]+','+sel[1]+')'+(n?'':' — nothing to do here')) : 'at selected tile';
 const npc = sel? state.npcs.find(x=>x.x==sel[0]&&x.y==sel[1]) : null;
 const sb=document.getElementById('saybox'); sb.style.display=npc?'flex':'none';
 if(npc){sb.dataset.npc=npc.id;document.getElementById('sayto').textContent='say to '+npc.name+':';}
 const lg=document.getElementById('log');lg.textContent=acc.slice(-60).join('\\n');lg.scrollTop=lg.scrollHeight;
}
async function refresh(st){state=st||await (await fetch('/api/state'+(curZ===null?'':'?z='+curZ))).json();(state.narrative||[]).forEach(l=>acc.push(l));if(state.over)acc.push('*** You have died — press restart. ***');render();}
async function refreshZ(){state=await (await fetch('/api/state'+(curZ===null?'':'?z='+curZ))).json();state.narrative=[];render();}
async function post(u,o){return (await fetch(u,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(o||{})})).json();}
async function doStep(i){refresh(await post('/api/step',{i}));}
async function doSay(){const t=document.getElementById('utter').value.trim();if(!t)return;document.getElementById('utter').value='';acc.push('> ('+(document.getElementById('tone').value||'neutral')+') '+t);refresh(await post('/api/say',{npc:document.getElementById('saybox').dataset.npc,text:t,tone:document.getElementById('tone').value||null}));}
async function doNew(){acc=[];sel=null;refresh(await post('/api/new',{}));}
async function doAct(){const t=document.getElementById('doit').value.trim();if(!t)return;document.getElementById('doit').value='';acc.push('> '+t);refresh(await post('/api/act',{text:t}));}
document.getElementById('utter').addEventListener('keydown',e=>{if(e.key==='Enter')doSay();});
document.getElementById('doit').addEventListener('keydown',e=>{if(e.key==='Enter')doAct();});
window.onload=()=>refresh();
</script></body></html>"""


class _Game:
    def __init__(self, picker=None):
        self.picker = picker
        self.reset()

    def reset(self):
        self.world, self.player = scene()
        self.last = []

    def state(self, z=None):
        w, p = self.world, self.player
        W, H, Z = w.dims
        z = p.pos[2] if z is None else max(0, min(Z - 1, int(z)))
        opts = engine.affordance_menu(w, p)
        reach = dict(engine.npcs_in_reach(w, p))
        return {
            "grid": engine.render_grid(w, z),
            "z": z, "zmax": Z,
            "player": list(p.pos),
            "cells": {f"{x},{y}": engine.cell_info(w, (x, y, z)) for x in range(W) for y in range(H)},
            "options": [{"i": i, "label": o.label,
                         "target": list(o.target[:2]) if o.target and o.target[2] == z else None}
                        for i, o in enumerate(opts)],
            "npcs": [{"id": i, "name": nm, "x": w.entities[i].pos[0], "y": w.entities[i].pos[1]}
                     for i, nm in reach.items() if w.entities[i].pos[2] == z],
            "over": engine.is_over(w),
            "narrative": self.last,
        }

    def step(self, i):
        opts = engine.affordance_menu(self.world, self.player)
        self.last = engine.step(self.world, opts[i])["narrative"] if 0 <= i < len(opts) else []

    def say(self, npc, text, tone=None):
        self.last = engine.say(self.world, npc, text, tone)["narrative"]

    def act(self, text):
        self.last = engine.free_text(self.world, self.player, text, self.picker)["narrative"]


def serve(port: int = 8000, brain: str = "mock", voice: str = "none"):
    from . import composer
    social.use_brain(brain)
    social.use_voice(voice)
    if brain == "torch":                    # load once, up front — not on the first click
        print("warming up decision model...")
        from .llm import get_lm
        get_lm().load()
    if voice != "none" and social.VOICE is not None:
        print("warming up voice model...")
        social.VOICE._load()
    picker = composer.TorchPicker() if brain == "torch" else composer.MockPicker()
    if brain == "torch":                    # the LM also picks consequences that FIT the fiction
        from . import checks
        checks.SELECTOR = lambda scene, utt, q, opts: picker.pick(scene, utt, q, opts)
    game = _Game(picker=picker)

    class H(BaseHTTPRequestHandler):
        def _send(self, code, body, ctype="application/json"):
            data = body if isinstance(body, bytes) else body.encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self._send(200, PAGE, "text/html; charset=utf-8")
            elif self.path.startswith("/api/state"):
                z = None
                if "z=" in self.path:
                    z = self.path.split("z=")[1].split("&")[0]
                st = game.state(z)
                st["narrative"] = []        # only POST responses carry fresh events (no replays)
                self._send(200, json.dumps(st))
            else:
                self._send(404, "{}")

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0) or 0)
            body = json.loads(self.rfile.read(n) or b"{}")
            if self.path == "/api/step":
                game.step(int(body.get("i", -1)))
            elif self.path == "/api/say":
                game.say(body["npc"], str(body.get("text", "")), body.get("tone") or None)
            elif self.path == "/api/act":
                game.act(str(body.get("text", "")))
            elif self.path == "/api/new":
                game.reset()
            else:
                return self._send(404, "{}")
            self._send(200, json.dumps(game.state()))

        def log_message(self, *a):
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", port), H)
    print(f"Mark-1 web UI → http://127.0.0.1:{port}   (brain={brain}, voice={voice})")
    srv.serve_forever()
