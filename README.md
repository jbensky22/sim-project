# sim-project
Much has been made of the 'bullpen revolution' over the past couple of years. Andrew Miller and Chris Devinski represent relievers on the forefront of the revolution, on teams at the forefront of  innovation. The Astros rotuinley use Devinski in the middle innings, and for multiple innings. Devinski, a skilled pitcher who could close on many teams, provides a bridge to the 'high leverage relievers' , and affords the Astros bullpen flexibility that is usually unseen with conventional bullpen management. Conventional bullpen management calls for a closer, a couple of set up men, a LOOGY or 2, and some lower leverage relievers. With a lead, the closer always pitches the ninth, the setup men pitch the seventh and eight, and LOOGY's are sometimes called upon for one batter. The opitmal effieiceny of this sytem occurs when the starter pitches at least 6 innings. With this model, the sp transitions smoothly into brief outings by the 'high leverage' relievers who should close out the game. However, as pointed out in recent articles by Russel Carleton at baseball prospectus, starting pitchers often fail to reach the sixth inning. Starters are pitching less every year, and real evidence is being found of a penalty for going through the order 3 times. 

Carelton, and others at baseball prospectus, have examined different possible rotations constructs that would make starters more efficient. Rotation ideas include .......... What I propose is something slightly different and more radical. 

With the rise of the modern day bullpens, many times are often able to shorten games. If the yankees get to the seventh inning with a lead, robertson, betances, chapman are tough to say the least. 

My method proposes a group of 7 pitchers capable of handling a starters workload.  The goal of these 7 pitchers is to maximize the amount of games the team enters  the 7th inning with a lead. The tradional model would have the 5 best pitchers pitch every fifth day, and if they are unable to complete 6, they are relived. Instead, my system proposes a tandem-type method where decisions are made based on the leverage of the game in different innings. If your goal is to reach the 7th inning with the lead, having a good pitcher available to bridge the gap and conserve the lead makes intuitive sense. Furthermore, this system calls for less ip per outing, which, in theory, will allow the better pitchers to effect more games. 

Simulation Specifics

I created a Monte Carlo simulation in Python with the ultimate  goal of seeing if two teams, comprised of the same exact pitchers, may achieve different results using different pitching management  strategies. 

I started by gathering pitcher data from Fangraphs. I got ERA data for starters and relievers  who qualified over the past three years. Then, using random sampling in python, I randomly sampled 150 times from the starter data. These 150 samples represent the 150 starters in my simlation, and each starter was placed on one of 30 arbitrary teams. I did the same for relievers, generating 7 per team. 

Now, with 30 teams, each of differing skill levels,  I could simulate a season.   While each team had high leverage relievers, for the sake of this model, I only looked at the  5 starters and the 2 worst relievers ( the mop-up men) for each team. I also insured that the 2 mop-up men always had worse ERA's than any starter on their team. 

 I first simulated the season using traditional rotation management. Each pitcher went as far as he could, and was removed  based on simplistic criteria that relied on the amount of runs he had given up and the amount of innings he had pitched. Of course, more goes into deciding weather to pull a sp, but for the sake of this simulation, I kept the criteria simple. Innings were simulated all at once, with the amount of runs determined by a possion random number generator which incorporated the pitchers era.  No offesnse was used in the run generation, only the pitching talent level was considered. After each game is simulated through six innings, the winner and loser is recorded; in the event of a tie, the away team gets credited with a win. 

For the second simulation, the starting rotation is made up of the number 2, 3,4,and 5 starters. The ace never starts!!  

In the simulation, the starter always pitches at least 3 innings, regardless of his performance. He goes out for the fourth inning only if he's given up less than 3 runs and he pitched less than 5 innings in his previous start.  He goes out for the fifth inning only if he's given up less than 2 runs and he pitched less than 4 innings in his previous start.  He goes out for the sixth inning only if the ace and the two mop-up men are not available. If the starter is pulled, either the ace or one of the mop-up men is brought into the game, depending on the leverage of the situation. 

The criteria above is one instance in which the starter is removed due to rest or runs given up. There is another instance in which the starter can be removed. If the Leverage Index is greater than 1.1 heading into the 5th or 6th innings, and the ace is sufficiently rested (determined by other criteria), the ace will be brought in. 

Inevitable Criticisms
 
I believe this simulation hones in on one important question. How valuable is a pitcher throwing 6 innings of two run baseball, but doing so once every 5 days? Is he more valuable pitching 3 times a week, two innings at a time, with his runs allowed more likely spread over three games? Many will say that a starter who can go deep into games and keep the your team in games is invaluable. It keeps the bullpen fresh and gives your team a great chance to win.  I don't necessarily disagree. But I think the notion that this is the most efficient way to manage a whole rotation is short sighted. A majority of starters struggle to get through six. This system requires them to only get through 3 or 4 innings max. 

Some still might not be sold. By forgoing starts from best starters, some argue you are giving up a great chance to win. Once again, I don't disagree. However, by having the best starters available to come in in the third, fourth, fifth, or sixth, your creating the opportunity to win games you might otherwise lose had you let the inferior pitcher remain in the game. The idea of using the best pitcher when it matters most also overlooks the fact that there will be less situations that 'matter' if you have poor pitchers giving up 4 runs in the first two innings more often. 

And finally, the question of if pitchers can pitch on 4 days rest if they are only going 3-5 innings at a time is important to consider.  Russel Carelton showed that pitchers going on 3 days of rest are unaffected in their performance. 

Results

For each of the thirty teams in my fictional league, I simulated 100 seasons where each team used the new pitching strategy, and every other team used the old pitching strategy. On average, teams added .6 wins a season. The max wins added was 1.32, the min wins added was -.666. There does seem to be a very slight advantage to be had from saving your ace for the big moments. 

Conclusion

The future of the five man rotation is in question. As teams and analysts explore alternate strategies, the question posed by this article will certainly be raised. Through this analysis, it seems the value of a  start by an ace once a week can be matched, and beaten, by 2 or 3 separate two inning high leverage outings by that same ace.  Furthermore, it is known that pitchers moving to the bullpen perform better because they are able to exert more energy per outing, since they pitch less than starters. A question remains, however, on weather pitching less per outing but pitching in more games  allows pitchers to exert more energy per outing, despite pitching the same amount over a given time, say a week.

I hypothesize that pitching less per outing   but pitching in more games would allow a pitcher to perform slightly better in each outing than they would pitching a full 5 or 6 innings. However, the effects of pitching a couple innings every other day or every three days could catch up to a pitcher over the course of a season. Then again, it could be beneficial to the pitcher by allowing them more opportunity to work on their craft. Like stated previously, Russel Carelton has showed that pitch count in the previous game effects next game performance more than the amount of rest. But a strategy such as the one proposed here calls for very short rest and short outings throughout the year, something that hasnt been seen in decades. Thus, it is no doubt a risky strategy. But as the game changes, more pitchers will start to fill the role filled by the ace in my simulation. Pitching 2-3 innings multiple times a week is on the horizon.   I propose that teams look into  handing that role to the ace. 


The  practicality of this simulation is lost a bit in simplicity and in the unknown. Runs are modeled using a random number generator, and pitching changes are ruled by a small series of if and elif statments. Not to mention the simulation only allows pitchers to be subbed before an inning starts.  Certainly, real baseball is more complex. Regardless, I believe the simulation provides a framework to understand different pitching strategies. Future work could involve an examination of other pitcher management strategies as well as added complexity. 




