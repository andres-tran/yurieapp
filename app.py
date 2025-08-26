# app.py
import os
import base64
from io import BytesIO

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

# --- Load env (works locally; Streamlit Cloud can use st.secrets instead) ---
load_dotenv(override=False)

def _get_api_key() -> str | None:
    key = os.getenv("OPENAI_API_KEY")
    if key:
        return key
    # Streamlit Cloud: add OPENAI_API_KEY in App Secrets
    try:
        return st.secrets["OPENAI_API_KEY"]  # type: ignore[index]
    except Exception:
        return None

# --- Configure page ---
st.set_page_config(page_title="AI Chatbot (OpenAI + Streamlit)", page_icon="🤖")
st.title("🤖 AI Chatbot (OpenAI + Streamlit)")

# --- Sidebar controls ---
with st.sidebar:
    st.header("Settings")
    default_model = os.getenv("OPENAI_MODEL", "gpt-5")
    model = st.text_input("Model", value=default_model, help="Try gpt-5, gpt-4o, etc.")
    use_web = st.checkbox(
        "Use web search (preview)",
        value=os.getenv("ENABLE_WEB_SEARCH", "false").lower() == "true",
        help="Adds OpenAI's web_search_preview tool to ground answers on the web.",
    )
    sys_prompt = st.text_area(
        "System prompt",
        value="You are a helpful, concise assistant.",
        height=96,
        help="Applied each turn. (Responses API doesn't carry previous instructions automatically.)",
    )

    st.markdown("---")
    if st.button("🧹 New chat"):
        st.session_state.clear()
        st.rerun()

# --- Make sure we have an API key ---
api_key = _get_api_key()
if not api_key:
    st.error(
        "Missing OPENAI_API_KEY. Add it to a local .env or to Streamlit → App Secrets."
    )
    st.stop()

client = OpenAI(api_key=api_key)

# --- Session state ---
st.session_state.setdefault("messages", [])  # [{'role': 'user'|'assistant', 'content': str}]
st.session_state.setdefault("previous_response_id", None)

# --- Tabs: Chat + Image ---
tab_chat, tab_image = st.tabs(["💬 Chat", "🖼️ Image"])

# -------------------------------
# Chat tab
# -------------------------------
with tab_chat:
    # Display prior messages
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    # Chat input
    if prompt := st.chat_input("Ask me anything…"):
        # Show user turn
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Assistant turn (streamed)
        with st.chat_message("assistant"):
            placeholder = st.empty()
            acc_text = ""

            tools = [{"type": "web_search_preview"}] if use_web else []

            try:
                # Stream tokens from the Responses API
                # (Streaming events like response.output_text.delta are documented by OpenAI.)
                with client.responses.stream(
                    model=model,
                    instructions=sys_prompt,
                    input=prompt,  # only the new user turn; context comes via previous_response_id
                    tools=tools,   # optional web search tool
                    previous_response_id=st.session_state.get("previous_response_id"),
                ) as stream:
                    for event in stream:
                        if event.type == "response.output_text.delta":
                            acc_text += event.delta
                            # Render incrementally
                            placeholder.markdown(acc_text)
                        elif event.type == "response.error":
                            placeholder.error(str(event.error))

                    final = stream.get_final_response()
            except Exception as e:
                placeholder.error(f"OpenAI error: {e}")
                final = None

            # Persist and render the final text
            out_text = getattr(final, "output_text", None) or acc_text
            if out_text:
                placeholder.markdown(out_text)
                st.session_state.messages.append(
                    {"role": "assistant", "content": out_text}
                )
                # Save response id to keep conversation state across turns
                st.session_state["previous_response_id"] = getattr(final, "id", None)

    st.caption(
        "Tip: Toggle **Use web search** in the sidebar to ground answers with live web results."
    )

# -------------------------------
# Image tab
# -------------------------------
with tab_image:
    st.write("Generate images with streamed partial previews.")
    image_prompt = st.text_area(
        "Image prompt",
        "Draw a gorgeous image of a river made of white owl feathers, snaking its way through a serene winter landscape",
        height=100,
    )
    n_partials = st.number_input(
        "Partial images to stream", min_value=0, max_value=4, value=2, step=1
    )
    img_model = os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-1")

    if st.button("Generate image"):
        gallery = []
        final_bytes = None
        spot = st.empty()

        try:
            # Stream partial images as they are generated; show them live
            stream = client.images.generate(
                prompt=image_prompt,
                model=img_model,
                stream=True,
                partial_images=int(n_partials),
            )

            for event in stream:
                # Partial frames during generation (documented by OpenAI as 'partial_image')
                if getattr(event, "type", "") == "image_generation.partial_image":
                    img_b64 = event.b64_json
                    img_bytes = base64.b64decode(img_b64)
                    gallery.append(img_bytes)
                    spot.image(img_bytes, caption="Partial preview", use_container_width=True)

                # Final image event (name may evolve; we handle both gracefully)
                elif getattr(event, "type", "") in ("image_generation.image", "image.image"):
                    img_b64 = event.b64_json
                    final_bytes = base64.b64decode(img_b64)
                    spot.image(final_bytes, caption="Final image", use_container_width=True)

            # Fallback: if the final event shape changes, show the last partial
            if final_bytes is None and gallery:
                final_bytes = gallery[-1]
                spot.image(final_bytes, caption="Final (from last partial)", use_container_width=True)

            if final_bytes:
                st.download_button(
                    "Download image",
                    data=final_bytes,
                    file_name="generated.png",
                    mime="image/png",
                )

        except Exception as e:
            st.error(f"Image generation error: {e}")
            st.stop()
