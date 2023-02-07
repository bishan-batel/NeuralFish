using System.Collections.Generic;
using System.Data.SqlTypes;
using System.Linq;
using Godot;

namespace NeuralFish.AI;

public class Fish : Node2D
{
	const float IdleFoodConsumptionPerSecond = 0.1f;
	const float MovingFoodConsumptionPerSecond = 0.5f;
	const float MatingFoodCost = 0.5f;
	const float IdleMatingFoodConsumptionPerSecond = 0.15f;
	const int BrainInputs = 11;
	const int BrainOutputs = 4;

	static readonly int[] BrainHiddenLayers =
	{
		10, 8, 6
	};

	Vector2 _velocity = Vector2.Zero;
	Color _color = Colors.White;
	float _foodLevel = 0.2f;

	NeuralNetwork _brain;

	public override void _Ready()
	{
		_brain ??= new NeuralNetwork(BrainInputs, BrainOutputs, BrainHiddenLayers);
		AddToGroup("fish");
	}

	public override void _PhysicsProcess(float delta)
	{
		Fish closestFish = GetClosestFish();
		Vector2 closestFishPos = GlobalPosition - (closestFish?.GlobalPosition ?? GlobalPosition);

		float[] outputs = _brain.FeedForward(new[]
		{
			closestFishPos.x, closestFishPos.y, Rotation, _velocity.x, _velocity.y, 0, 0, 0, _foodLevel,
		});

		float turn = Mathf.Lerp(-1, 1, outputs[0]);
		float thrust = Mathf.Lerp(-1, 1, outputs[1]);
		bool mate = outputs[2] > 0.5;
		bool eat = outputs[3] > 0.5;

		// turn
		Rotation += turn * delta * 2;

		// thrust
		_velocity += new Vector2(Mathf.Cos(Rotation), Mathf.Sin(Rotation)) * thrust * delta * 100;

		// eat
		if (eat)
		{
			_foodLevel += IdleFoodConsumptionPerSecond * delta;

			// get closest food
			// TODO
		}

		// mate
		if (mate)
		{
			_foodLevel -= IdleMatingFoodConsumptionPerSecond * delta;

			if (closestFish != null) Mate(closestFish);
		}
	}

	void Mate(Fish other)
	{
	}

	Fish GetClosestFish() => GetTree()
			.GetNodesInGroup("fish")
			.Cast<Fish>()
			.OrderBy(f => f.GlobalPosition.DistanceTo(GlobalPosition))
			.FirstOrDefault();
}