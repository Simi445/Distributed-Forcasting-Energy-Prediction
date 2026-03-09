import { CardSmall } from './MLCard';

export default function IdleComponent({changeState, models})
  {
    return(
      <>
      <h1 className="text-3xl font-bold mb-4 ">Model Section</h1>
      <div className="grid gap-6 md:grid-cols-1 ">
        {models.map((model) => (
          <CardSmall key={model.id} cardTitle={model.cardTitle} onClick={() => changeState(model.cardTitle)}/>
        ))}
      </div>
      </>
    );
  }