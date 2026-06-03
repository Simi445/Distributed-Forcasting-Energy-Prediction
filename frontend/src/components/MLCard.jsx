import { Button } from "@/components/ui/button"
import {
  Card,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"

export function CardSmall({cardTitle, onClick}) {
  return (
    <Card size="sm" className="mx-auto w-full max-w-sm rounded-lg">
      <CardHeader>
        <CardTitle>{cardTitle}</CardTitle>
      </CardHeader>
      <CardFooter>
        <Button variant="outline" size="sm" className="w-full" onClick={onClick}>
          See Page
        </Button>
      </CardFooter>
    </Card>
  )
}
