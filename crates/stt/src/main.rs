fn main() {
    voice_engine_stt::run_helper(voice_engine_protocol::HelperConfig {
        env_prefix: "VOICE_ENGINE",
    });
}
