import { avatar } from "@/lib/format";
import { cn } from "@/lib/utils";

/** Deterministic, tinted initials avatar for a sender name/email. */
export function Avatar({
  name,
  className,
}: {
  name: string;
  className?: string;
}) {
  const { initials, colorClass } = avatar(name);
  return (
    <div
      className={cn(
        "flex size-9 shrink-0 items-center justify-center rounded-full text-[11px] font-semibold",
        colorClass,
        className,
      )}
      aria-hidden="true"
    >
      {initials}
    </div>
  );
}
