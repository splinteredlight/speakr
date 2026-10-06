"""
Shared speaker identification service.

Provides LLM-based speaker identification from transcript context,
used by both the web UI (recordings.py) and REST API (api_v1.py).
"""

import os
import re
import json
from flask import current_app


def _known_full_names(user_id, participants):
    """Candidate full names: the recording's participants first, then the user's saved speakers."""
    from src.models import Speaker

    names, seen = [], set()
    sources = [p.strip() for p in re.split(r'[,;\n]', participants or '')]
    try:
        sources += [s.name for s in Speaker.query.filter_by(user_id=user_id).order_by(Speaker.use_count.desc()).all()]
    except Exception as e:
        current_app.logger.warning(f"[Auto-Identify] Could not load saved speakers: {e}")
    for name in sources:
        name = re.sub(r'\s+', ' ', name or '').strip()
        if name and name.lower() not in seen:
            seen.add(name.lower())
            names.append(name)
    return names


def _expand_to_full_name(name, participant_names, saved_names):
    """Expand a bare first name to a full name when exactly one known person matches."""
    if not name or ' ' in name:
        return name
    for pool in (participant_names, saved_names):
        matches = {n for n in pool if ' ' in n and n.split(' ')[0].lower() == name.lower()}
        if len(matches) == 1:
            return matches.pop()
        if matches:
            return name  # ambiguous: leave as-is rather than guess
    return name


def identify_speakers_from_transcript(transcription_data, user_id, participants=None):
    """
    Identify speakers in a transcription using an LLM.

    Args:
        transcription_data: List of transcript segments (already parsed JSON).
        user_id: Current user's ID (for token tracking).
        participants: Optional recording participants string; used (with saved
            speakers) to resolve first names to full names.

    Returns:
        dict mapping original speaker labels to identified names.
        Values are empty string "" for unidentified speakers.

    Raises:
        ValueError: If LLM API key is not configured.
        Exception: On LLM call failure.
    """
    from src.services.llm import call_llm_completion
    from src.utils import safe_json_loads
    from src.models import SystemSetting

    # Extract unique speakers in order of appearance
    seen_speakers = set()
    unique_speakers = []
    for segment in transcription_data:
        speaker = segment.get('speaker')
        if speaker and speaker not in seen_speakers:
            seen_speakers.add(speaker)
            unique_speakers.append(speaker)

    if not unique_speakers:
        return {}

    # Normalize all labels to SPEAKER_XX format for the LLM
    speaker_to_label = {}
    for idx, speaker in enumerate(unique_speakers):
        speaker_to_label[speaker] = f'SPEAKER_{str(idx).zfill(2)}'

    # Create temporary transcript with normalized labels
    formatted_lines = []
    for segment in transcription_data:
        original_speaker = segment.get('speaker')
        label = speaker_to_label.get(original_speaker, 'Unknown Speaker')
        sentence = segment.get('sentence', '')
        formatted_lines.append(f"[{label}]: {sentence}")
    formatted_transcription = "\n".join(formatted_lines)

    speaker_labels = list(speaker_to_label.values())

    current_app.logger.info(f"[Auto-Identify] Formatted transcript (first 500 chars): {formatted_transcription[:500]}")
    current_app.logger.info(f"[Auto-Identify] Speaker labels: {speaker_labels}")

    # Apply configurable transcript length limit
    transcript_limit = SystemSetting.get_setting('transcript_length_limit', 30000)
    if transcript_limit == -1:
        transcript_text = formatted_transcription
    else:
        transcript_text = formatted_transcription[:transcript_limit]

    participant_names = [p for p in (re.sub(r'\s+', ' ', x).strip() for x in re.split(r'[,;\n]', participants or '')) if p]
    known_names = _known_full_names(user_id, participants)
    saved_names = [n for n in known_names if n not in participant_names]
    known_block = ""
    if known_names:
        known_block = (
            "\n\nKnown people (the listed participants of this meeting come first, then other people this user "
            "has recorded before). When a speaker is referred to by first name or nickname and it matches one of "
            "these people, answer with that person's FULL name exactly as written here:\n"
            + "\n".join(f"- {n}" for n in known_names[:150])
        )

    prompt = f"""Analyze the following conversation transcript and identify the names of the speakers based on the context and content of their dialogue.

The speakers that need to be identified are: {', '.join(speaker_labels)}

Look for clues in the conversation such as:
- Names mentioned by other speakers when addressing someone
- Self-introductions or references to their own name
- Context clues about roles, relationships, or positions
- Any direct mentions of names in the dialogue

Here is the complete conversation transcript:

{transcript_text}{known_block}

Based on the conversation above, identify the most likely real names for the speakers. Pay close attention to how speakers address each other and any names that are mentioned in the dialogue.

Respond with a single JSON object where keys are the speaker labels (e.g., "SPEAKER_01") and values are the identified full names. If a name cannot be determined from the conversation context, use an empty string "".

Example format:
{{
  "SPEAKER_01": "Jane Smith",
  "SPEAKER_03": "Bob Johnson",
  "SPEAKER_05": ""
}}

JSON Response:
"""

    current_app.logger.info("[Auto-Identify] Calling LLM")

    use_schema = os.environ.get('AUTO_IDENTIFY_RESPONSE_SCHEMA', '').strip() in ('1', 'true', 'yes')
    system_msg = (
        "You are an expert in analyzing conversation transcripts to identify speakers "
        "based on contextual clues in the dialogue. Analyze the conversation carefully "
        "to find names mentioned when speakers address each other or introduce themselves. "
        "Your response must be a single, valid JSON object containing only the requested "
        "speaker identifications."
    )

    response_content = None
    if use_schema:
        # Build JSON schema response format with constrained keys
        schema_properties = {label: {"type": "string"} for label in speaker_labels}
        schema_response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "speaker_identification",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": schema_properties,
                    "required": speaker_labels,
                    "additionalProperties": False
                }
            }
        }
        schema_prompt = prompt + f"\n\nIMPORTANT: Your JSON response must contain exactly these keys: {', '.join(speaker_labels)}"
        try:
            current_app.logger.info("[Auto-Identify] Trying json_schema response format")
            completion = call_llm_completion(
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": schema_prompt}
                ],
                temperature=0.2,
                response_format=schema_response_format,
                user_id=user_id,
                operation_type='speaker_identification'
            )
            response_content = completion.choices[0].message.content
            current_app.logger.info(f"[Auto-Identify] LLM Raw Response (schema mode): {response_content}")
        except Exception as schema_err:
            current_app.logger.warning(f"[Auto-Identify] json_schema mode failed, falling back to json_object: {schema_err}")
            response_content = None

    if response_content is None:
        completion = call_llm_completion(
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            response_format={"type": "json_object"},
            user_id=user_id,
            operation_type='speaker_identification'
        )
        response_content = completion.choices[0].message.content
        current_app.logger.info(f"[Auto-Identify] LLM Raw Response: {response_content}")

    identified_map = safe_json_loads(response_content, {})
    current_app.logger.info(f"[Auto-Identify] Parsed identified_map: {identified_map}")

    # --- Sanitize identified_map ---
    identified_map = _sanitize_identified_map(identified_map, speaker_labels)
    current_app.logger.info(f"[Auto-Identify] Sanitized identified_map: {identified_map}")

    # Resolve bare first names to full names from participants / saved speakers
    for label, name in identified_map.items():
        identified_map[label] = _expand_to_full_name(name, participant_names, saved_names)
    current_app.logger.info(f"[Auto-Identify] Full-name-resolved map: {identified_map}")

    # Map back to original speaker labels
    final_speaker_map = {}
    for original_speaker, temp_label in speaker_to_label.items():
        if temp_label in identified_map:
            final_speaker_map[original_speaker] = identified_map[temp_label]

    current_app.logger.info(f"[Auto-Identify] Final speaker_map: {final_speaker_map}")
    return final_speaker_map


def _sanitize_identified_map(identified_map, speaker_labels):
    """
    Clean up LLM output: handle inverted maps, strip commentary,
    clear placeholders, etc.
    """
    speaker_label_re = re.compile(r'^SPEAKER_\d{2}$')

    # Detect inverted map ({name: "SPEAKER_XX"}) and flip it
    if identified_map and all(
        speaker_label_re.match(str(v)) for v in identified_map.values() if v
    ) and not any(speaker_label_re.match(str(k)) for k in identified_map.keys()):
        current_app.logger.warning("[Auto-Identify] Detected inverted map, flipping keys/values")
        identified_map = {v: k for k, v in identified_map.items() if v}

    sanitized = {}
    for speaker_label, identified_name in identified_map.items():
        # Skip entries whose key isn't a valid SPEAKER_XX label
        if not speaker_label_re.match(str(speaker_label)):
            continue
        if not identified_name or not isinstance(identified_name, str):
            sanitized[speaker_label] = ""
            continue

        name = identified_name.strip()

        # Clear generic placeholders
        if name.lower() in ["unknown", "n/a", "not available", "unclear", "unidentified", ""]:
            sanitized[speaker_label] = ""
            continue

        # Clear label-to-label entries (e.g. "SPEAKER_01": "SPEAKER_02")
        if speaker_label_re.match(name):
            sanitized[speaker_label] = ""
            continue

        # Strip parenthetical content: "John (the host)" -> "John"
        name = re.sub(r'\s*\([^)]*\)', '', name).strip()

        # Take first name segment before comma, semicolon, or slash
        name = re.split(r'[,;/]', name)[0].strip()

        # Collapse whitespace
        name = re.sub(r'\s+', ' ', name)

        # Final check: if result still matches SPEAKER_XX, clear it
        if speaker_label_re.match(name) or not name:
            sanitized[speaker_label] = ""
            continue

        sanitized[speaker_label] = name

    return sanitized
