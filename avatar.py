"""
Simple talking avatar component for Streamlit.
Uses inline SVG + CSS animation (free, no external assets).
Called by app.py for display in the chat header.
"""

def render_avatar():
    """Return HTML string with an animated talking face."""
    return """
    <style>
    @keyframes breathe {
        0%, 100% { transform: translateY(0) scale(1); opacity: 0.9; }
        50% { transform: translateY(-4px) scale(1.05); opacity: 1; }
    }
    @keyframes mouthOpen {
        0%, 100% { height: 6px; border-radius: 10px; }
        50% { height: 14px; border-radius: 12px 12px 4px 4px; }
    }
    .simple-avatar-wrap { text-align: center; margin: 10px 0 18px 0; }
    .avatar-face {
        width: 70px; height: 70px; border-radius: 50%;
        background: linear-gradient(135deg, #8b5cf6, #38bdf8);
        box-shadow: 0 4px 30px rgba(56,189,248,0.25), 0 0 0 2px rgba(56,189,248,0.2);
        display: inline-block; position: relative;
        animation: breathe 3s ease-in-out infinite;
    }
    .avatar-mouth {
        width: 28px; height: 8px;
        background: rgba(255,255,255,0.9);
        border-radius: 10px;
        position: absolute; bottom: 22px; left: 21px;
        animation: mouthOpen 1.2s ease-in-out infinite;
        box-shadow: inset 0 1px 4px rgba(0,0,0,0.15);
    }
    .avatar-eyes {
        position: absolute; top: 22px; left: 16px;
        width: 38px; height: 10px; display: flex; justify-content: space-between;
    }
    .avatar-eye {
        width: 10px; height: 10px; border-radius: 50%;
        background: rgba(255,255,255,0.95);
        box-shadow: inset 0 1px 3px rgba(0,0,0,0.2);
    }
    </style>
    <div class="simple-avatar-wrap">
        <div class="avatar-face">
            <div class="avatar-eyes">
                <div class="avatar-eye"></div>
                <div class="avatar-eye"></div>
            </div>
            <div class="avatar-mouth"></div>
        </div>
    </div>
    """
