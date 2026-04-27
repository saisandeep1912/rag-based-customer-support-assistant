from app.application import run_support_query


def main() -> None:
    print("Customer Support Agent")
    question = input("Ask your support question: ").strip()
    if not question:
        print("No question provided.")
        return

    try:
        response = run_support_query(question)
    except RuntimeError as exc:
        print(f"Error: {exc}")
        return

    print()
    print(f"Route: {response.route}")
    print(f"Confidence: {response.confidence:.2f}")
    print(f"Reasoning: {response.reasoning}")
    print(f"Answer: {response.answer or 'No answer generated.'}")

    if response.escalation_ticket_id:
        print(f"Escalation Ticket ID: {response.escalation_ticket_id}")


if __name__ == "__main__":
    main()
