"""
モックアップHTMLテンプレート。

モデルが画面構成とデータを決め、このテンプレートに埋め込む。
モデル自身がHTMLを書く必要がないので、品質が安定する。
"""

DESIGN_TOKENS = {
    "material": {
        "primary": "#6750A4",
        "primary_light": "#E8DEF8",
        "surface": "#FFFBFE",
        "on_surface": "#1C1B1F",
        "secondary": "#625B71",
        "error": "#B3261E",
        "success": "#386A20",
        "radius": "16px",
        "shadow": "0 2px 8px rgba(0,0,0,0.12)",
        "font": "'Roboto', 'Noto Sans JP', sans-serif",
    },
    "ios": {
        "primary": "#007AFF",
        "primary_light": "#E3F2FD",
        "surface": "#F2F2F7",
        "on_surface": "#000000",
        "secondary": "#8E8E93",
        "error": "#FF3B30",
        "success": "#34C759",
        "radius": "12px",
        "shadow": "none",
        "font": "system-ui, -apple-system, sans-serif",
    },
    "fluent": {
        "primary": "#0078D4",
        "primary_light": "#DEECF9",
        "surface": "#F3F3F3",
        "on_surface": "#323130",
        "secondary": "#605E5C",
        "error": "#D13438",
        "success": "#107C10",
        "radius": "4px",
        "shadow": "0 1px 4px rgba(0,0,0,0.08)",
        "font": "'Segoe UI', system-ui, sans-serif",
    },
    "simple": {
        "primary": "#333333",
        "primary_light": "#F0F0F0",
        "surface": "#FFFFFF",
        "on_surface": "#333333",
        "secondary": "#888888",
        "error": "#CC0000",
        "success": "#228B22",
        "radius": "8px",
        "shadow": "none",
        "font": "system-ui, sans-serif",
    },
}


def build_mockup_html(screens: list[dict], design: str = "material", app_name: str = "App") -> str:
    """
    screens: [
        {
            "title": "ホーム",
            "type": "card_list" | "detail" | "form" | "list" | "profile" | "empty_state",
            "cards": [{"title": "...", "subtitle": "...", "badge": "..."}],  # card_list用
            "content": "...",  # detail用
            "fields": [{"label": "...", "placeholder": "..."}],  # form用
            "items": [{"text": "...", "right": "..."}],  # list用
            "stats": [{"label": "...", "value": "..."}],  # profile用
            "message": "...",  # empty_state用
            "buttons": [{"label": "...", "color": "primary|secondary|error"}],
        }
    ]
    """
    t = DESIGN_TOKENS.get(design, DESIGN_TOKENS["material"])

    sections_html = []
    for i, screen in enumerate(screens):
        screen_type = screen.get("type", "card_list")
        title = screen.get("title", f"画面{i+1}")
        body = _render_screen_body(screen, t)
        buttons_html = _render_buttons(screen.get("buttons", []), t)

        sections_html.append(f"""
    <section class="screen">
      <div class="screen-header">{title}</div>
      <div class="screen-body">
        {body}
        {buttons_html}
      </div>
    </section>""")

    nav_items = [s.get("title", f"画面{i+1}") for i, s in enumerate(screens[:5])]
    nav_html = "".join(
        f'<div class="nav-item">{"📱🏠⚙️📋👤"[i % 5]}<br><span>{name[:4]}</span></div>'
        for i, name in enumerate(nav_items)
    )

    return f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{app_name} - モックアップ</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ background: #eee; font-family: {t['font']}; }}
.phone-frame {{
  width: 390px; margin: 20px auto; background: {t['surface']};
  border: 2px solid #ccc; border-radius: 32px; overflow: hidden;
  box-shadow: 0 4px 20px rgba(0,0,0,0.15);
}}
.status-bar {{
  background: {t['on_surface']}; color: #fff; padding: 8px 20px;
  font-size: 12px; display: flex; justify-content: space-between;
}}
.app-header {{
  background: {t['primary']}; color: #fff; padding: 14px 20px;
  font-size: 18px; font-weight: bold; text-align: center;
}}
.screen {{ border-bottom: 8px solid {t['surface']}; }}
.screen-header {{
  background: {t['primary_light']}; color: {t['on_surface']};
  padding: 10px 16px; font-size: 13px; font-weight: bold;
  letter-spacing: 1px; text-transform: uppercase;
}}
.screen-body {{ padding: 16px; }}
.card {{
  background: #fff; border-radius: {t['radius']};
  box-shadow: {t['shadow']}; padding: 14px 16px; margin-bottom: 12px;
  border: 1px solid rgba(0,0,0,0.06);
}}
.card-title {{ font-size: 16px; font-weight: bold; color: {t['on_surface']}; }}
.card-subtitle {{ font-size: 13px; color: {t['secondary']}; margin-top: 4px; }}
.card-badge {{
  display: inline-block; background: {t['primary']}; color: #fff;
  font-size: 11px; padding: 2px 8px; border-radius: 10px; margin-top: 6px;
}}
.detail-text {{ font-size: 14px; color: {t['on_surface']}; line-height: 1.7; }}
.form-field {{
  margin-bottom: 14px;
}}
.form-label {{ font-size: 12px; color: {t['secondary']}; margin-bottom: 4px; }}
.form-input {{
  width: 100%; padding: 10px 12px; border: 1px solid #ddd;
  border-radius: {t['radius']}; font-size: 14px; background: #fafafa;
}}
.list-item {{
  display: flex; justify-content: space-between; align-items: center;
  padding: 12px 0; border-bottom: 1px solid #f0f0f0;
}}
.list-text {{ font-size: 14px; color: {t['on_surface']}; }}
.list-right {{ font-size: 13px; color: {t['secondary']}; }}
.stat-row {{ display: flex; gap: 12px; margin-bottom: 12px; }}
.stat-box {{
  flex: 1; background: {t['primary_light']}; border-radius: {t['radius']};
  padding: 12px; text-align: center;
}}
.stat-value {{ font-size: 22px; font-weight: bold; color: {t['primary']}; }}
.stat-label {{ font-size: 11px; color: {t['secondary']}; margin-top: 2px; }}
.empty-msg {{
  text-align: center; padding: 40px 20px; color: {t['secondary']};
  font-size: 14px;
}}
.empty-icon {{ font-size: 48px; margin-bottom: 12px; }}
.btn {{
  display: inline-block; padding: 10px 20px; border-radius: {t['radius']};
  font-size: 14px; font-weight: bold; cursor: pointer; border: none;
  margin: 4px 4px 4px 0; text-align: center;
}}
.btn-primary {{ background: {t['primary']}; color: #fff; }}
.btn-secondary {{ background: {t['primary_light']}; color: {t['primary']}; }}
.btn-error {{ background: {t['error']}; color: #fff; }}
.btn-row {{ margin-top: 12px; }}
.bottom-nav {{
  display: flex; justify-content: space-around; background: #fff;
  border-top: 1px solid #e0e0e0; padding: 8px 0;
}}
.nav-item {{
  text-align: center; font-size: 18px; color: {t['secondary']};
  line-height: 1.2;
}}
.nav-item span {{ font-size: 10px; }}
</style>
</head>
<body>
<div class="phone-frame">
  <div class="status-bar"><span>9:41</span><span>📶 🔋</span></div>
  <div class="app-header">{app_name}</div>
  {"".join(sections_html)}
  <div class="bottom-nav">{nav_html}</div>
</div>
</body>
</html>"""


def _render_screen_body(screen: dict, t: dict) -> str:
    screen_type = screen.get("type", "card_list")

    if screen_type == "card_list":
        cards = screen.get("cards", [])
        html = ""
        for card in cards:
            badge = f'<div class="card-badge">{card["badge"]}</div>' if card.get("badge") else ""
            html += f"""
        <div class="card">
          <div class="card-title">{card.get('title', '')}</div>
          <div class="card-subtitle">{card.get('subtitle', '')}</div>
          {badge}
        </div>"""
        return html

    if screen_type == "detail":
        content = screen.get("content", "")
        return f'<div class="detail-text">{content}</div>'

    if screen_type == "form":
        fields = screen.get("fields", [])
        html = ""
        for field in fields:
            html += f"""
        <div class="form-field">
          <div class="form-label">{field.get('label', '')}</div>
          <input class="form-input" placeholder="{field.get('placeholder', '')}" disabled>
        </div>"""
        return html

    if screen_type == "list":
        items = screen.get("items", [])
        html = ""
        for item in items:
            html += f"""
        <div class="list-item">
          <span class="list-text">{item.get('text', '')}</span>
          <span class="list-right">{item.get('right', '')}</span>
        </div>"""
        return html

    if screen_type == "profile":
        stats = screen.get("stats", [])
        html = '<div class="stat-row">'
        for stat in stats:
            html += f"""
          <div class="stat-box">
            <div class="stat-value">{stat.get('value', '')}</div>
            <div class="stat-label">{stat.get('label', '')}</div>
          </div>"""
        html += '</div>'
        return html

    if screen_type == "empty_state":
        msg = screen.get("message", "データがありません")
        return f'<div class="empty-msg"><div class="empty-icon">📭</div>{msg}</div>'

    return ""


def _render_buttons(buttons: list, t: dict) -> str:
    if not buttons:
        return ""
    html = '<div class="btn-row">'
    for btn in buttons:
        color_class = f"btn-{btn.get('color', 'primary')}"
        html += f'<button class="btn {color_class}">{btn.get("label", "ボタン")}</button>'
    html += '</div>'
    return html
