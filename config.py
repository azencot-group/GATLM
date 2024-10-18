dataset_defaults = {
    'ETTm1': {


        'data_path': 'ETTm1.csv',
        'task_id': 'ETTm1',

        'data': 'ETTm1',
        'features': 'M',
        'seq_len': 96,
        'label_len': 48,

        'e_layers': 2,
        'd_layers': 1,
        'factor': 3,
        'enc_in': 7,
        'dec_in': 7,
        'c_out': 7,
        'des': 'Exp',
        'd_model': 512,

    },
    'ETTh1': {


        'data_path': 'ETTh1.csv',
        'task_id': 'ETTh1',

        'data': 'ETTh1',
        'features': 'M',
        'seq_len': 96,
        'label_len': 48,

        'e_layers': 2,
        'd_layers': 1,
        'factor': 3,
        'enc_in': 7,
        'dec_in': 7,
        'c_out': 7,
        'des': 'Exp',
        'd_model': 512,

    },
    'ETTm2': {


        'data_path': 'ETTm2.csv',
        'task_id': 'ETTm2',

        'data': 'ETTm2',
        'features': 'M',
        'seq_len': 96,
        'label_len': 48,

        'e_layers': 2,
        'd_layers': 1,
        'factor': 3,
        'enc_in': 7,
        'dec_in': 7,
        'c_out': 7,
        'des': 'Exp',
        'd_model': 512,

    },
    'ETTh2': {

        'data_path': 'ETTh2.csv',
        'task_id': 'ETTh2',

        'data': 'ETTh2',
        'features': 'M',
        'seq_len': 96,
        'label_len': 48,

        'e_layers': 2,
        'd_layers': 1,
        'factor': 3,
        'enc_in': 7,
        'dec_in': 7,
        'c_out': 7,
        'des': 'Exp',
        'd_model': 512,

    },
    'ECL': {


        'data_path': 'electricity.csv',
        'task_id': 'ECL',

        'data': 'custom',
        'features': 'M',
        'seq_len': 96,
        'label_len': 48,

        'e_layers': 2,
        'd_layers': 1,
        'factor': 3,
        'enc_in': 321,
        'dec_in': 321,
        'c_out': 321,
        'des': 'Exp',

    },
    'Traffic': {


        'data_path': 'traffic.csv',
        'task_id': 'traffic',

        'data': 'custom',
        'features': 'M',
        'seq_len': 96,
        'label_len': 48,

        'e_layers': 2,
        'd_layers': 1,
        'factor': 3,
        'enc_in': 862,
        'dec_in': 862,
        'c_out': 862,
        'des': 'Exp',
    },
    'Weather': {


        'data_path': 'weather.csv',
        'task_id': 'weather',
        'data': 'custom',
        'features': 'M',
        'seq_len': 96,
        'label_len': 48,
        'e_layers': 2,
        'd_layers': 1,
        'factor': 3,
        'enc_in': 21,
        'dec_in': 21,
        'c_out': 21,
        'des': 'Exp',
    }
}
