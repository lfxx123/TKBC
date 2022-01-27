# TKBC-IMR

An Interpretable Multi-hop Reasoning (IMR) model for temporal KG forecasting.
![image](https://user-images.githubusercontent.com/49195527/143236285-d4c3d02e-4d4c-439e-987e-93e93cffeed3.png)


<!-- This is the code of paper Interpretable Multi-hop Reasoning for Forecasting Future Links on Temporal Knowledge Graphs. Zongwei Liang, Junan Yang, Keju Huang, Hui Liu. -->

## Dependencies
<!-- 库版本 -->
        pip install -r requirements.txt

## Results
* Dataset in [TITer](https://github.com/JHL-HUST/TITer/)
![image](https://user-images.githubusercontent.com/49195527/143236179-683bdfb2-abe2-406f-9128-6d72a6d9fce0.png)

* The result
![image](https://user-images.githubusercontent.com/49195527/143236076-a01827ee-a42c-4355-ab08-0ebc8902f140.png)


## Reproduce the Results

        bash ./run_ICEW18.sh
        bash ./run_ICEWS14.sh
        bash ./run_WIKI.sh
        bash ./run_YAGO.sh


## Citation
If you find this code useful, please consider citing the following paper.

    @inproceedings{
    anonymous2022interpretable,
    title={Interpretable Multi-hop Reasoning for Forecasting Future Links on Temporal Knowledge Graphs},
    author={Anonymous},
    booktitle={Submitted to The Tenth International Conference on Learning Representations },
    year={2022},
    url={https://openreview.net/forum?id=OQo6Tuyo0ih},
    note={under review}
    }
If you have any questions, please email me.

## Acknowledgement
We refer to the code of [xERTE](https://github.com/TemporalKGTeam/xERTE). Thanks for their contributions.
