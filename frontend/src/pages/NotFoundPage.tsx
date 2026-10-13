import { Link } from "react-router-dom";
import { Compass } from "lucide-react";
import { EmptyState } from "../components/ui";
export default function NotFoundPage() {
  return (
    <EmptyState
      title="Page not found"
      message="This route is not part of the dashboard. The navigation lists every available page."
      icon={<Compass className="h-6 w-6" />}
      action={
        <Link to="/" className="btn-primary">
          Back to the overview
        </Link>
      }
    />
  );
}
