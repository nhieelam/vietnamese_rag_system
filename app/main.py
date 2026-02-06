from app.services import SessionService
from app.ui import (
    apply_custom_styles,
    render_sidebar,
    render_chat_messages,
    render_chat_input,
)


def main():
    SessionService.initialize()
    
    apply_custom_styles()
    
    render_sidebar()
    render_chat_messages()
    render_chat_input()


if __name__ == "__main__":
    main()