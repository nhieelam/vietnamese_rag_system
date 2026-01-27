
from app.services import initialize_session_state
from app.ui import (
    apply_custom_styles,
    render_sidebar,
    render_chat_messages,
    render_chat_input,
)



def main():

    apply_custom_styles()
    
    initialize_session_state()
    
    render_sidebar()
    
    render_chat_messages()
    
    render_chat_input()


if __name__ == "__main__":
    main()